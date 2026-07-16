"""
Web dashboard service — Aviation HUD in the browser.

Subscribes to:
  Channel.PERCEPTION_FRAME   — frames → JPEG over WebSocket (binary)
  Channel.REASONING_SCENE    — tracks + vehicle state + lead (JSON, ≤10 Hz)
  Channel.REASONING_EVENT    — alerts for the ticker (JSON)
  Channel.VOICE_UTTERANCE    — Cruze's voice line (JSON)

FastAPI + uvicorn run in-process as asyncio tasks. fastapi/uvicorn/cv2 are
optional: missing deps log a warning and the service idles (graceful
degradation, same pattern as the cv2 HUD).

NOTE: no `from __future__ import annotations` here. PEP 563 stringifies
annotations, and FastAPI then cannot resolve the lazily-imported WebSocket
type on the /ws endpoint — it falls back to treating the parameter as a
required query field and rejects every handshake with 1008/403.
"""

import asyncio
import json
import logging
import pathlib
import time
from typing import TYPE_CHECKING, Any

from cruze.core.bus import Channel, EventBus
from cruze.core.types import DrivingEvent, Frame, Scene, Utterance
from cruze.hmi import serialize

if TYPE_CHECKING:
    from cruze.core.config import Config

logger = logging.getLogger(__name__)

_WEB_DIR = pathlib.Path(__file__).parent / "web"

# Scene JSON cap — full track lists at camera rate would saturate the socket;
# 10 Hz is smooth for gauges and overlays.
_SCENE_MAX_HZ = 10.0


class ClientHub:
    """
    Outbound fan-out to connected browsers with drop-oldest semantics.

    Each client gets its own queue of ("bin", bytes) / ("txt", str) items;
    a slow client loses old frames instead of stalling the pumps.
    """

    def __init__(self, maxsize: int = 16) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._maxsize = maxsize

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(self._maxsize)
        self._clients.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def broadcast(self, item: tuple[str, Any]) -> None:
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(item)

    @property
    def client_count(self) -> int:
        return len(self._clients)


def build_app(web_dir: pathlib.Path, hub: ClientHub):
    """Build the FastAPI app. Imports fastapi lazily so the module imports clean."""
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Cruze Dashboard")

    @app.get("/")
    async def index():
        return FileResponse(web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = hub.register()
        try:
            while True:
                kind, payload = await q.get()
                if kind == "bin":
                    await ws.send_bytes(payload)
                else:
                    await ws.send_text(payload)
        except Exception:
            pass  # client disconnected (WebSocketDisconnect or transport error)
        finally:
            hub.unregister(q)

    return app


class DashboardService:
    def __init__(self, cfg: "Config", bus: EventBus) -> None:
        self._cfg = cfg
        self._bus = bus
        self._hub = ClientHub()
        self._server = None
        self._running = False

    async def run(self) -> None:
        self._running = True
        hmi = self._cfg.hmi

        if not hmi.enabled or hmi.backend not in ("web", "both"):
            logger.info("Dashboard: disabled (backend=%s)", hmi.backend)
            while self._running:
                await asyncio.sleep(1.0)
            return

        try:
            import uvicorn
        except ImportError:
            logger.warning(
                "Dashboard: fastapi/uvicorn not installed — disabled. "
                "Install with: pip install cruze[dashboard]"
            )
            while self._running:
                await asyncio.sleep(1.0)
            return

        app = build_app(_WEB_DIR, self._hub)
        config = uvicorn.Config(
            app, host=hmi.web_host, port=hmi.web_port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)

        pumps = [
            asyncio.create_task(self._pump_frames(), name="dash-frames"),
            asyncio.create_task(self._pump_scenes(), name="dash-scenes"),
            asyncio.create_task(self._pump_events(), name="dash-events"),
            asyncio.create_task(self._pump_utterances(), name="dash-voice"),
        ]
        logger.info("Dashboard: serving on http://%s:%d", hmi.web_host, hmi.web_port)
        failed = False
        try:
            await self._server.serve()
        except (SystemExit, OSError) as exc:
            # uvicorn calls sys.exit(1) when it cannot bind (typically the
            # port is held by another Cruze instance). A dead dashboard must
            # not take the co-pilot down — log and idle instead.
            logger.warning(
                "Dashboard: could not serve on %s:%d (%s) — disabled. "
                "Is another Cruze instance already running?",
                hmi.web_host, hmi.web_port, exc,
            )
            failed = True
        finally:
            for t in pumps:
                t.cancel()
        if failed:
            while self._running:
                await asyncio.sleep(1.0)

    async def stop(self) -> None:
        self._running = False
        if self._server is not None:
            self._server.should_exit = True

    async def _pump_frames(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.warning("Dashboard: opencv not installed — video stream disabled")
            return
        queue = self._bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=2)
        loop = asyncio.get_running_loop()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.hmi.jpeg_quality]
        while self._running:
            try:
                frame: Frame = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if self._hub.client_count == 0:
                continue  # don't burn CPU encoding for nobody
            ok, buf = await loop.run_in_executor(
                None, cv2.imencode, ".jpg", frame.image, encode_params
            )
            if ok:
                self._hub.broadcast(("bin", buf.tobytes()))

    async def _pump_scenes(self) -> None:
        queue = self._bus.subscribe(Channel.REASONING_SCENE, maxsize=4)
        last_sent = 0.0
        while self._running:
            try:
                scene: Scene = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            now = time.monotonic()
            if now - last_sent < 1.0 / _SCENE_MAX_HZ:
                continue
            last_sent = now
            self._hub.broadcast(("txt", json.dumps(serialize.scene_to_dict(scene))))

    async def _pump_events(self) -> None:
        queue = self._bus.subscribe(Channel.REASONING_EVENT, maxsize=8)
        while self._running:
            try:
                ev: DrivingEvent = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            self._hub.broadcast(("txt", json.dumps(serialize.event_to_dict(ev))))

    async def _pump_utterances(self) -> None:
        queue = self._bus.subscribe(Channel.VOICE_UTTERANCE, maxsize=8)
        while self._running:
            try:
                utt: Utterance = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            self._hub.broadcast(("txt", json.dumps(serialize.utterance_to_dict(utt))))
