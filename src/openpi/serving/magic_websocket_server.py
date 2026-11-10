from __future__ import annotations

import http
import json
import logging

import websockets.asyncio.server as _server

from openpi.serving import websocket_policy_server as _base

logger = logging.getLogger(__name__)


class MagicWebsocketPolicyServer(_base.WebsocketPolicyServer):
    """WebSocket policy service with health, session reset, and status endpoints."""

    def __init__(self, *args, reset_on_connect: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_on_connect = reset_on_connect

    # ---------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    def _process_request(self, connection: _server.ServerConnection, request: _server.Request):
        path = request.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")

        if path == "/reset":
            try:
                self._policy.reset(reason="http_endpoint")
            except TypeError:
                self._policy.reset()
            logger.info("[MagicWebsocketPolicyServer] HTTP /reset received; session reset")
            body = json.dumps({"ok": True, "action": "reset"}, ensure_ascii=False)
            return connection.respond(http.HTTPStatus.OK, body + "\n")

        if path == "/status":
            status = self._policy.status() if hasattr(self._policy, "status") else {}
            body = json.dumps({"ok": True, "status": status}, ensure_ascii=False, default=str)
            return connection.respond(http.HTTPStatus.OK, body + "\n")

        return None

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    # ---------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    async def _handler(self, websocket: _server.ServerConnection):
        if self._reset_on_connect:
            try:
                self._policy.reset(reason="connect")
            except TypeError:
                self._policy.reset()
            logger.info("[MagicWebsocketPolicyServer] New connection; session reset")
        await super()._handler(websocket)
