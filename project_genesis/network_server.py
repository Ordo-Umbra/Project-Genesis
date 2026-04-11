"""Lightweight WebSocket server for remote monitoring and control.

Run alongside the simulation loop to expose a JSON-based API:

- ``get_state``          → world summary
- ``get_chunk``          → voxel data for a specific chunk
- ``get_agent_view``     → perception data for a specific agent
- ``send_action``        → queue an action for an agent
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import websockets.asyncio.server

try:
    import websockets
    import websockets.asyncio.server as ws_server
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]
    ws_server = None  # type: ignore[assignment]

from .engine import GenesisEngine


class NetworkServer:
    """WebSocket server that reads from (and writes to) a GenesisEngine."""

    def __init__(
        self,
        engine: GenesisEngine,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        if websockets is None:
            raise RuntimeError("Install the 'websockets' package to use NetworkServer")
        self.engine = engine
        self.host = host
        self.port = port
        self._clients: set[Any] = set()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_get_state(self) -> dict:
        return {"type": "state", "data": self.engine.get_world_summary()}

    def _handle_get_chunk(self, payload: dict) -> dict | bytes:
        cx = int(payload.get("x", 0))
        cy = int(payload.get("y", 0))
        cz = int(payload.get("z", 0))
        cm = self.engine.chunk_manager
        if not (0 <= cx < cm.grid_shape[0] and 0 <= cy < cm.grid_shape[1] and 0 <= cz < cm.grid_shape[2]):
            return {"type": "error", "message": "chunk index out of range"}
        data = cm.get_chunk_data(self.engine.get_field_snapshot(), cx, cy, cz)
        flat = data.astype(np.float32).tobytes()
        # Return shape header + raw float32 bytes for efficiency.
        header = struct.pack("!3I", *data.shape)
        return {"type": "chunk", "x": cx, "y": cy, "z": cz, "shape": list(data.shape), "data_b64": _bytes_to_b64(header + flat)}

    def _handle_get_agent_view(self, payload: dict) -> dict:
        agent_id = str(payload.get("agent_id", ""))
        for agent in self.engine.agents:
            if agent.agent_id == agent_id:
                perception = agent.get_perception(
                    self.engine.get_field_snapshot(),
                    agents=self.engine.agents,
                    beta=self.engine.BETA,
                )
                return {"type": "agent_view", "data": perception}
        return {"type": "error", "message": f"agent '{agent_id}' not found"}

    def _handle_send_action(self, payload: dict) -> dict:
        agent_id = str(payload.get("agent_id", ""))
        action = payload.get("action", {})
        ok = self.engine.queue_agent_action(agent_id, action)
        return {"type": "action_ack", "agent_id": agent_id, "queued": ok}

    # ------------------------------------------------------------------
    # WebSocket connection handler
    # ------------------------------------------------------------------

    async def _handler(self, ws: Any) -> None:
        self._clients.add(ws)
        try:
            async for raw_message in ws:
                try:
                    message = json.loads(raw_message)
                except (json.JSONDecodeError, TypeError):
                    await ws.send(json.dumps({"type": "error", "message": "invalid JSON"}))
                    continue

                cmd = message.get("command", "")
                if cmd == "get_state":
                    response = self._handle_get_state()
                elif cmd == "get_chunk":
                    response = self._handle_get_chunk(message)
                elif cmd == "get_agent_view":
                    response = self._handle_get_agent_view(message)
                elif cmd == "send_action":
                    response = self._handle_send_action(message)
                else:
                    response = {"type": "error", "message": f"unknown command: {cmd}"}

                await ws.send(json.dumps(response, default=_json_default))
        finally:
            self._clients.discard(ws)

    # ------------------------------------------------------------------
    # Push notifications
    # ------------------------------------------------------------------

    async def _broadcast(self, message: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(message, default=_json_default)
        disconnected = set()
        for client in self._clients:
            try:
                await client.send(data)
            except Exception:
                disconnected.add(client)
        self._clients -= disconnected

    def notify_chunk_updated(self, cx: int, cy: int, cz: int) -> None:
        """Push a ``chunk_updated`` event to all connected clients."""
        if self._loop is None:
            return
        msg = {"type": "chunk_updated", "x": cx, "y": cy, "z": cz}
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        async with ws_server.serve(self._handler, self.host, self.port):
            await self._stop_event.wait()

    def start(self) -> None:
        """Start the WebSocket server in a background daemon thread."""

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request the server to stop."""
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Handle numpy types when serializing to JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _bytes_to_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")
