"""Tests for the alert state machine (countdown, cancel, cooldown, car gating)."""

import asyncio
import time

import pytest

from src.alerts import AlertManager, AlertState


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_manager(clock=None, sent=None, ok=True, **kw):
    sent = sent if sent is not None else []

    async def dispatcher(alert):
        sent.append(alert)
        return ok

    defaults = dict(mode="human", dispatcher=dispatcher, countdown_s=0.05, cooldown_s=60)
    defaults.update(kw)
    return AlertManager(clock=clock or Clock(), **defaults), sent


def run(coro):
    return asyncio.run(coro)


def test_event_creates_pending_alert():
    m, _ = make_manager()
    alert, reason = m.on_event("FALL", 0.9, 12.9, 77.5, None)
    assert reason == "pending" and alert.state is AlertState.PENDING
    assert m.pending is alert


def test_countdown_dispatches():
    async def scenario():
        m, sent = make_manager(clock=time.monotonic)
        alert, _ = m.on_event("FALL", 0.9)
        events = []

        async def sink(e):
            events.append(e)
        await m.start_countdown(sink)
        return alert, sent, events, m

    alert, sent, events, m = run(scenario())
    assert sent == [alert] and alert.state is AlertState.DISPATCHED
    assert events[0]["event"] == "alert_dispatched"
    assert m.pending is None


def test_cancel_prevents_dispatch():
    async def scenario():
        m, sent = make_manager(clock=time.monotonic, countdown_s=0.2)
        alert, _ = m.on_event("FALL", 0.9)
        task = m.start_countdown()
        await asyncio.sleep(0.01)
        assert m.cancel(alert.id) is alert
        await asyncio.sleep(0.3)
        return alert, sent, task

    alert, sent, task = run(scenario())
    assert sent == [] and alert.state is AlertState.CANCELLED and task.done()


def test_cancel_wrong_id_is_ignored():
    m, _ = make_manager()
    alert, _ = m.on_event("FALL", 0.9)
    assert m.cancel(alert.id + 99) is None
    assert m.pending is alert


def test_cooldown_and_single_pending():
    clock = Clock()
    m, _ = make_manager(clock=clock)
    a, _ = m.on_event("FALL", 0.9)
    assert m.on_event("FALL", 0.9) == (None, "alert_already_pending")
    m.cancel()
    assert m.on_event("FALL", 0.9) == (None, "cooldown")
    clock.t += 61
    b, reason = m.on_event("FALL", 0.9)
    assert reason == "pending" and b.id != a.id


def test_dispatch_failure_is_reported():
    async def scenario():
        m, _ = make_manager(ok=False)
        alert, _ = m.on_event("FALL", 0.9)
        await m.dispatch_now(alert)
        return alert

    assert run(scenario()).state is AlertState.FAILED


def test_dispatcher_exception_does_not_propagate():
    async def boom(alert):
        raise RuntimeError("twilio down")

    async def scenario():
        m = AlertManager("human", boom, clock=Clock())
        alert, _ = m.on_event("FALL", 0.9)
        await m.dispatch_now(alert)
        return alert

    assert run(scenario()).state is AlertState.FAILED


def test_late_cancel_cannot_abort_dispatch():
    async def scenario():
        m, sent = make_manager()
        alert, _ = m.on_event("FALL", 0.9)
        await m.dispatch_now(alert)
        return m.cancel(alert.id), sent

    cancelled, sent = run(scenario())
    assert cancelled is None and len(sent) == 1


def test_detach_without_dispatch_on_disconnect_cancels():
    m, _ = make_manager(dispatch_on_disconnect=False)
    alert, _ = m.on_event("FALL", 0.9)
    m.detach()
    assert alert.state is AlertState.CANCELLED


@pytest.mark.parametrize("speeds, expected", [
    ([0.0, 2.0, 1.0], "vehicle_not_moving"),   # parked car, phone dropped
    ([60.0, 40.0, 0.0], "pending"),            # was moving, now stopped → plausible crash
    ([], "pending"),                           # no GPS → fail safe
])
def test_car_speed_gating(speeds, expected):
    m, _ = make_manager(mode="car", min_recent_speed_kmh=15.0)
    for s in speeds:
        m.observe(s)
    m.observe(None)
    assert m.on_event("CRASH", 0.97)[1] == expected


def test_speed_history_expires():
    clock = Clock()
    m, _ = make_manager(clock=clock, mode="car", min_recent_speed_kmh=15.0, speed_lookback_s=30)
    m.observe(80.0)
    clock.t += 31
    m.observe(0.0)
    assert m.speed.max_recent == 0.0
