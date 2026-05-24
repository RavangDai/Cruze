"""Tests for the event bus — no ML deps required."""

import asyncio
import pytest

from cruze.core.bus import EventBus, Channel


@pytest.mark.asyncio
async def test_publish_subscribe_basic():
    bus = EventBus()
    q = bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=4)
    await bus.publish(Channel.PERCEPTION_FRAME, "hello")
    msg = await asyncio.wait_for(q.get(), timeout=1.0)
    assert msg == "hello"


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_message():
    bus = EventBus()
    q1 = bus.subscribe(Channel.REASONING_EVENT, maxsize=4)
    q2 = bus.subscribe(Channel.REASONING_EVENT, maxsize=4)
    await bus.publish(Channel.REASONING_EVENT, "event")
    assert await asyncio.wait_for(q1.get(), timeout=1.0) == "event"
    assert await asyncio.wait_for(q2.get(), timeout=1.0) == "event"


@pytest.mark.asyncio
async def test_drop_oldest_on_overflow():
    bus = EventBus()
    q = bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=2)
    # Fill beyond capacity — oldest should be evicted.
    await bus.publish(Channel.PERCEPTION_FRAME, 1)
    await bus.publish(Channel.PERCEPTION_FRAME, 2)
    await bus.publish(Channel.PERCEPTION_FRAME, 3)  # 1 should be dropped
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert 1 not in items
    assert 2 in items and 3 in items


@pytest.mark.asyncio
async def test_no_cross_channel_bleed():
    bus = EventBus()
    qa = bus.subscribe(Channel.PERCEPTION_FRAME, maxsize=4)
    qb = bus.subscribe(Channel.REASONING_EVENT, maxsize=4)
    await bus.publish(Channel.PERCEPTION_FRAME, "frame_msg")
    assert qa.qsize() == 1
    assert qb.qsize() == 0


def test_subscriber_count():
    bus = EventBus()
    assert bus.subscriber_count(Channel.VOICE_WAKE) == 0
    bus.subscribe(Channel.VOICE_WAKE)
    bus.subscribe(Channel.VOICE_WAKE)
    assert bus.subscriber_count(Channel.VOICE_WAKE) == 2
