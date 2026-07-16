"""Dashboard service tests. FastAPI-dependent tests are skipped when the
optional [dashboard] extra is not installed — the core suite must pass with
stdlib + numpy only."""

import asyncio

import pytest

from cruze.core.config import HMIConfig
from cruze.hmi.webapp import ClientHub


def test_hmi_web_defaults():
    cfg = HMIConfig()
    assert cfg.backend == "web"
    assert cfg.web_host == "127.0.0.1"
    assert cfg.web_port == 8484
    assert 1 <= cfg.jpeg_quality <= 100
    assert cfg.show_masks is True
    assert cfg.show_lanes is True


@pytest.mark.asyncio
async def test_hub_broadcast_reaches_all_clients():
    hub = ClientHub(maxsize=4)
    q1, q2 = hub.register(), hub.register()
    hub.broadcast(("txt", "hello"))
    assert q1.get_nowait() == ("txt", "hello")
    assert q2.get_nowait() == ("txt", "hello")


@pytest.mark.asyncio
async def test_hub_drop_oldest_when_client_slow():
    hub = ClientHub(maxsize=2)
    q = hub.register()
    for i in range(4):
        hub.broadcast(("bin", i))
    assert q.qsize() == 2
    # Oldest items (0, 1) were dropped.
    assert q.get_nowait() == ("bin", 2)
    assert q.get_nowait() == ("bin", 3)


@pytest.mark.asyncio
async def test_hub_unregister_stops_delivery():
    hub = ClientHub()
    q = hub.register()
    hub.unregister(q)
    hub.broadcast(("txt", "x"))
    assert q.qsize() == 0
    assert hub.client_count == 0


def test_index_page_served():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # required by starlette's TestClient
    from fastapi.testclient import TestClient

    from cruze.hmi.webapp import _WEB_DIR, ClientHub, build_app

    client = TestClient(build_app(_WEB_DIR, ClientHub()))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "CRUZE" in resp.text


def test_static_frontend_served():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from cruze.hmi.webapp import _WEB_DIR, ClientHub, build_app

    client = TestClient(build_app(_WEB_DIR, ClientHub()))
    for name in ("app.js", "render.js", "style.css"):
        assert client.get(f"/static/{name}").status_code == 200, name


@pytest.mark.asyncio
async def test_dashboard_degrades_when_port_already_in_use():
    """Regression: uvicorn calls sys.exit(1) when it cannot bind; the
    SystemExit propagated out of DashboardService.run() and took down the
    whole Cruze process. A dead dashboard must idle, not crash the co-pilot."""
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import socket

    from cruze.core.bus import EventBus
    from cruze.core.config import Config
    from cruze.hmi.webapp import DashboardService

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        cfg = Config()
        cfg.hmi.backend = "web"
        cfg.hmi.web_port = port
        svc = DashboardService(cfg, EventBus())
        task = asyncio.ensure_future(svc.run())
        await asyncio.sleep(0.5)  # give uvicorn time to fail its bind
        assert not task.done(), (
            f"DashboardService died instead of idling: {task.exception()}"
        )
        await svc.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, SystemExit):
            pass
    finally:
        blocker.close()


def test_websocket_accepts_and_delivers_both_kinds():
    """Regression: with PEP 563 annotations + lazily imported WebSocket,
    FastAPI mistook the ws param for a required query field and rejected
    every handshake with 1008 (surfaced as HTTP 403 through uvicorn)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from cruze.hmi.webapp import _WEB_DIR, ClientHub, build_app

    hub = ClientHub()
    client = TestClient(build_app(_WEB_DIR, hub))
    with client.websocket_connect("/ws") as ws:
        hub.broadcast(("txt", '{"type":"ping"}'))
        assert ws.receive_text() == '{"type":"ping"}'
        hub.broadcast(("bin", b"\xff\xd8fakejpeg"))
        assert ws.receive_bytes() == b"\xff\xd8fakejpeg"
