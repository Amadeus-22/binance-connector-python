"""Fault-injection tests against a real local WebSocket / HTTP server.

Each test drives the client through a failure a production connection meets
(server shutdown notice, dropped socket, rotation) and checks what the caller
of an in-flight request observes.
"""

import asyncio
import json
import ssl
import threading
import time

import pytest
import requests
from aiohttp import web, WSMsgType
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from binance_common.configuration import (
    ConfigurationRestAPI,
    ConfigurationWebSocketAPI,
)
from binance_common.constants import WebsocketMode
from binance_common.models import WebsocketApiUserDataEndpoints
from binance_common.utils import send_request
from binance_common.websocket import WebSocketAPIBase

# Well below the 20 s per-request wait, well above a local round trip.
PROMPT = 3


class FakeWebSocketAPI:
    """A local WebSocket API server whose behaviour each test scripts."""

    def __init__(self, on_request, on_connect=None):
        self.on_request = on_request
        self.on_connect = on_connect
        self.sockets = []
        self.received = []
        self.clients = []

    async def _handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets.append(ws)
        if self.on_connect:
            asyncio.ensure_future(self.on_connect(ws, len(self.sockets)))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                self.received.append((len(self.sockets), data))
                await self.on_request(ws, data)
        return ws

    async def __aenter__(self):
        app = web.Application()
        app.router.add_get("/ws", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0, shutdown_timeout=1)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/ws"
        return self

    async def __aexit__(self, *exc):
        for client in self.clients:
            if client.connections:
                await client.close_connection()
            elif client.session is not None:
                await client.session.close()
        for ws in self.sockets:
            await ws.close()
        await self._runner.cleanup()


async def reply(ws, data):
    await ws.send_str(
        json.dumps({"id": data["id"], "status": 200, "result": {}, "rateLimits": []})
    )


def make_client(server):
    configuration = ConfigurationWebSocketAPI(
        api_key="key",
        api_secret="secret",
        stream_url=server.url,
        mode=WebsocketMode.SINGLE,
        reconnect_delay=0,
    )
    client = WebSocketAPIBase(
        configuration,
        WebsocketApiUserDataEndpoints(
            user_data_stream_subscribe="userDataStream.subscribe",
            user_data_stream_unsubscribe="userDataStream.unsubscribe",
            user_data_stream_logout="session.logout",
        ),
    )
    server.clients.append(client)
    return client


async def timed_send(client, method="order.place"):
    started = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.send_message({"method": method, "params": {}}), timeout=PROMPT
        )
    except asyncio.TimeoutError:
        return None, time.monotonic() - started
    return response.data(), time.monotonic() - started


class TestInFlightRequestsAcrossConnectionEvents:
    @pytest.mark.asyncio
    async def test_request_after_server_shutdown_gets_the_servers_answer(self):
        """Binance keeps serving a connection after `serverShutdown`.

        The server answers the order at once, on the connection it was sent on;
        the client must hand that answer to the caller instead of a timeout.
        """

        async def on_connect(ws, n):
            if n == 1:
                await asyncio.sleep(0.2)
                await ws.send_str(json.dumps({"event": {"e": "serverShutdown"}}))

        async with FakeWebSocketAPI(reply, on_connect) as server:
            client = make_client(server)
            await client.create_connection()
            await asyncio.sleep(0.6)

            data, elapsed = await timed_send(client)

            assert [d["method"] for _, d in server.received] == ["order.place"]
            assert data is not None, (
                f"server answered order.place at once, caller got nothing "
                f"after {elapsed:.1f}s"
            )
            assert data.get("status") == 200

    @pytest.mark.asyncio
    async def test_request_pending_when_connection_drops_fails_promptly(self):
        """A dropped connection must fail its pending requests, not leave them
        to the 20 s wait."""

        async def close_on_request(ws, data):
            await ws.close()

        async with FakeWebSocketAPI(close_on_request) as server:
            client = make_client(server)
            await client.create_connection()

            data, elapsed = await timed_send(client)

            assert data is not None, (
                f"connection closed at once, caller still waiting after "
                f"{elapsed:.1f}s"
            )
            assert "error" in data

    @pytest.mark.asyncio
    async def test_request_answered_during_rotation_reaches_the_caller(self):
        """The server answers on the old socket before the rotation closes it;
        that answer must not be discarded."""

        async def slow_reply(ws, data):
            await asyncio.sleep(0.1)
            await reply(ws, data)

        async with FakeWebSocketAPI(slow_reply) as server:
            client = make_client(server)
            await client.create_connection()
            old = client.connections[0]

            pending = asyncio.ensure_future(timed_send(client))
            await asyncio.sleep(0.02)
            await client.reconnect(old, client.configuration)

            data, elapsed = await pending
            assert data is not None, (
                f"request in flight at rotation got nothing after {elapsed:.1f}s"
            )


class _Ping(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    client_ports = set()

    def do_GET(self):
        _Ping.client_ports.add(self.client_address[1])
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestRestConnectionReuse:
    @pytest.mark.parametrize("https_agent", [None, ssl.create_default_context()])
    def test_keep_alive_reuses_one_connection(self, https_agent):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Ping)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _Ping.client_ports = set()
        try:
            configuration = ConfigurationRestAPI(
                api_key="key",
                api_secret="secret",
                base_path=f"http://127.0.0.1:{server.server_address[1]}",
                https_agent=https_agent,
            )
            session = requests.Session()
            for _ in range(10):
                send_request(session, configuration, "GET", "/api/v3/ping")

            assert configuration.keep_alive is True
            assert len(_Ping.client_ports) == 1, (
                f"10 keep-alive requests opened {len(_Ping.client_ports)} "
                f"TCP connections"
            )
        finally:
            server.shutdown()
