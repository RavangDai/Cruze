"""
Rule engine tests — verifies cooldowns, event firing, and multi-rule interaction.
"""

import asyncio
import time
import pytest

from cruze.core.bus import EventBus, Channel
from cruze.core.config import Config, ReasoningConfig
from cruze.core.types import BBox, EventLevel, ObjectClass, Scene, Track, VehicleState
from cruze.reasoning.events import EventEngine


def _make_config(**reasoning_overrides) -> Config:
    cfg = Config()
    for k, v in reasoning_overrides.items():
        setattr(cfg.reasoning, k, v)
    return cfg


def _fcw_scene() -> Scene:
    """Scene that triggers FCW: lead 5 m away, closing at 10 m/s, ego moving."""
    lead = Track(
        track_id=1,
        bbox=BBox(610, 300, 670, 400),
        cls=ObjectClass.CAR,
        distance_m=5.0,
        closing_speed_mps=10.0,
    )
    state = VehicleState(speed_mps=15.0, posted_speed_limit_mps=13.4)
    return Scene(tracks=(lead,), vehicle_state=state, lead_track=lead)


def _safe_scene() -> Scene:
    state = VehicleState(speed_mps=10.0, posted_speed_limit_mps=13.4)
    return Scene(vehicle_state=state)


@pytest.mark.asyncio
async def test_fcw_event_published():
    bus = EventBus()
    cfg = _make_config(event_cooldown_s=0.0)
    engine = EventEngine(cfg, bus)

    event_q = bus.subscribe(Channel.REASONING_EVENT, maxsize=10)

    task = asyncio.ensure_future(engine.run())
    await asyncio.sleep(0)  # let engine subscribe before we publish
    await bus.publish(Channel.REASONING_SCENE, _fcw_scene())

    # Drain events until we find an FCW or timeout.
    found_fcw = False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            event = await asyncio.wait_for(event_q.get(), timeout=0.1)
            if event.kind == "fcw":
                found_fcw = True
                assert event.level == EventLevel.CRITICAL
                break
        except asyncio.TimeoutError:
            break

    assert found_fcw, "FCW event was not published"
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_cooldown_suppresses_repeat():
    bus = EventBus()
    # Long cooldown: second identical scene must not re-fire any events.
    cfg = _make_config(event_cooldown_s=60.0)
    engine = EventEngine(cfg, bus)

    event_q = bus.subscribe(Channel.REASONING_EVENT, maxsize=20)

    task = asyncio.ensure_future(engine.run())
    await asyncio.sleep(0)  # let engine subscribe
    await bus.publish(Channel.REASONING_SCENE, _fcw_scene())
    await asyncio.sleep(0.05)
    first_count = event_q.qsize()
    assert first_count > 0, "Expected at least one event from first publish"

    await bus.publish(Channel.REASONING_SCENE, _fcw_scene())
    await asyncio.sleep(0.05)
    # No new events should have been added (all kinds on cooldown).
    assert event_q.qsize() == first_count, (
        f"Cooldown failed: expected {first_count} total events, got {event_q.qsize()}"
    )

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_no_event_for_safe_scene():
    bus = EventBus()
    cfg = _make_config(event_cooldown_s=0.0)
    engine = EventEngine(cfg, bus)

    event_q = bus.subscribe(Channel.REASONING_EVENT, maxsize=10)

    task = asyncio.ensure_future(engine.run())
    await bus.publish(Channel.REASONING_SCENE, _safe_scene())
    await asyncio.sleep(0.05)

    assert event_q.qsize() == 0
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_critical_event_has_shorter_cooldown():
    """CRITICAL events use half the normal cooldown."""
    bus = EventBus()
    # cooldown_s=0.0 so we can verify the engine fires; actual half-cooldown
    # logic is covered by inspecting EventEngine._evaluate directly.
    cfg = _make_config(event_cooldown_s=10.0)
    engine = EventEngine(cfg, bus)

    scene = _fcw_scene()
    # Manually call _evaluate twice with a simulated half-cooldown gap.
    engine._cooldowns["fcw"] = time.monotonic() - 5.01  # 5.01 s ago > half of 10 = 5
    events = engine._evaluate(scene)
    assert any(e.kind == "fcw" for e in events)

    # Less than half-cooldown gap → suppressed.
    engine._cooldowns["fcw"] = time.monotonic() - 4.0  # 4 s ago < 5
    events2 = engine._evaluate(scene)
    assert not any(e.kind == "fcw" for e in events2)


# --- IDM brake events ---

def _engine(**overrides) -> EventEngine:
    return EventEngine(_make_config(event_cooldown_s=0.0, **overrides), EventBus())


def _accel_scene(accel) -> Scene:
    return Scene(
        vehicle_state=VehicleState(speed_mps=15.0),
        required_accel_mps2=accel,
    )


def test_brake_hard_event_is_critical():
    events = _engine()._evaluate(_accel_scene(-6.0))
    hard = [e for e in events if e.kind == "brake_hard"]
    assert len(hard) == 1
    assert hard[0].level == EventLevel.CRITICAL
    assert hard[0].context["accel_mps2"] == -6.0
    assert not any(e.kind == "brake_advised" for e in events)


def test_brake_advised_event_is_warning():
    events = _engine()._evaluate(_accel_scene(-4.0))
    advised = [e for e in events if e.kind == "brake_advised"]
    assert len(advised) == 1
    assert advised[0].level == EventLevel.WARNING
    assert not any(e.kind == "brake_hard" for e in events)


def test_no_brake_events_in_comfort_zone():
    events = _engine()._evaluate(_accel_scene(-1.0))
    assert not any(e.kind in ("brake_hard", "brake_advised") for e in events)


# --- Cut-in detection ---

def _lead_scene(track_id, distance_m, ts) -> Scene:
    lead = Track(
        track_id=track_id,
        bbox=BBox(610, 300, 670, 400),
        cls=ObjectClass.CAR,
        distance_m=distance_m,
        closing_speed_mps=0.0,
    )
    return Scene(
        timestamp=ts,
        tracks=(lead,),
        vehicle_state=VehicleState(speed_mps=15.0),
        lead_track=lead,
    )


def test_cut_in_fires_on_closer_new_lead():
    engine = _engine()
    t0 = time.monotonic()
    engine._evaluate(_lead_scene(track_id=1, distance_m=40.0, ts=t0))
    events = engine._evaluate(_lead_scene(track_id=2, distance_m=25.0, ts=t0 + 0.1))
    cut = [e for e in events if e.kind == "cut_in"]
    assert len(cut) == 1
    assert cut[0].level == EventLevel.NOTICE
    assert cut[0].context["distance_m"] == 25.0


def test_cut_in_ignores_same_lead():
    engine = _engine()
    t0 = time.monotonic()
    engine._evaluate(_lead_scene(track_id=1, distance_m=40.0, ts=t0))
    events = engine._evaluate(_lead_scene(track_id=1, distance_m=25.0, ts=t0 + 0.1))
    assert not any(e.kind == "cut_in" for e in events)


def test_cut_in_ignores_stale_previous_scene():
    engine = _engine()
    t0 = time.monotonic()
    engine._evaluate(_lead_scene(track_id=1, distance_m=40.0, ts=t0))
    # 5 s gap between scenes → comparing leads is meaningless.
    events = engine._evaluate(_lead_scene(track_id=2, distance_m=25.0, ts=t0 + 5.0))
    assert not any(e.kind == "cut_in" for e in events)


# --- Lane departure ---

def _ldw_scene(cte_m) -> Scene:
    from cruze.core.types import Lanes

    return Scene(
        vehicle_state=VehicleState(speed_mps=15.0),
        lanes=Lanes(cte_m=cte_m),
    )


def test_lane_departure_right_side():
    events = _engine()._evaluate(_ldw_scene(cte_m=0.7))
    ldw = [e for e in events if e.kind == "lane_departure"]
    assert len(ldw) == 1
    assert ldw[0].level == EventLevel.WARNING
    assert ldw[0].context["side"] == "right"
    assert ldw[0].context["cte_m"] == 0.7


def test_lane_departure_left_side():
    events = _engine()._evaluate(_ldw_scene(cte_m=-0.7))
    ldw = [e for e in events if e.kind == "lane_departure"]
    assert len(ldw) == 1
    assert ldw[0].context["side"] == "left"
