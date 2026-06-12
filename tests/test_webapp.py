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
