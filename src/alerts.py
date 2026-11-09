"""
src/alerts.py
-------------
Per-session emergency alert policy.

Turning a model score into an SMS is the safety-critical part of the
system, so it lives here as a small, fully unit-tested state machine
instead of an ``if`` inside the WebSocket loop:

    IDLE ──event──► PENDING ──countdown expires──► DISPATCHED ─┐
                       │                                        │
                       └──user taps "I'm OK"──► CANCELLED ──────┴─► cooldown ─► IDLE

  - Countdown   : the user gets `countdown_s` seconds to cancel a false alarm.
  - Disconnect  : if the phone drops off mid-countdown (it may have been
                  destroyed in the crash) the alert is still dispatched.
  - Cooldown    : one incident produces one SMS, not one per window.
  - Car context : a "crash" while the vehicle was never moving (phone dropped
                  in a parked car) is suppressed when GPS speed is available.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_alert_ids = itertools.count(1)


class AlertState(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Alert:
    mode: str
    label: str
    confidence: float
    created_at: float
    deadline: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed_kmh: Optional[float] = None
    max_recent_speed_kmh: Optional[float] = None
    state: AlertState = AlertState.PENDING
    id: int = field(default_factory=lambda: next(_alert_ids))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d


class SpeedTracker:
    """Keeps the last `lookback_s` seconds of GPS speed readings."""

    def __init__(self, lookback_s: float, clock: Callable[[], float] = time.monotonic):
        self.lookback_s = lookback_s
        self._clock = clock
        self._history: deque[tuple[float, float]] = deque()

    def update(self, speed_kmh: Optional[float]) -> None:
        now = self._clock()
        if speed_kmh is not None:
            self._history.append((now, speed_kmh))
        while self._history and now - self._history[0][0] > self.lookback_s:
            self._history.popleft()

    @property
    def max_recent(self) -> Optional[float]:
        return max((s for _, s in self._history), default=None)


Dispatcher = Callable[[Alert], Awaitable[bool]]
EventSink = Callable[[dict], Awaitable[None]]


class AlertManager:
    """
    Parameters
    ----------
    mode : str
        "human" or "car".
    dispatcher : async (Alert) -> bool
        Sends the SOS (SMS / webhook). Must not block the event loop.
    countdown_s, cooldown_s : float
        See module docstring.
    min_recent_speed_kmh : float | None
        Car mode only – suppress crashes when GPS shows the car never moved.
    clock : () -> float
        Injectable monotonic clock (tests).
    """

    def __init__(
        self,
        mode: str,
        dispatcher: Dispatcher,
        countdown_s: float = 15.0,
        cooldown_s: float = 120.0,
        min_recent_speed_kmh: Optional[float] = None,
        speed_lookback_s: float = 30.0,
        dispatch_on_disconnect: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.mode = mode
        self.dispatcher = dispatcher
        self.countdown_s = countdown_s
        self.cooldown_s = cooldown_s
        self.min_recent_speed_kmh = min_recent_speed_kmh
        self.dispatch_on_disconnect = dispatch_on_disconnect
        self._clock = clock
        self.speed = SpeedTracker(speed_lookback_s, clock)

        self.pending: Optional[Alert] = None
        self.history: list[Alert] = []
        self._cooldown_until = 0.0
        self._task: Optional[asyncio.Task] = None
        self._sink: Optional[EventSink] = None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def observe(self, speed_kmh: Optional[float]) -> None:
        """Call for every incoming sample (keeps the GPS speed history fresh)."""
        self.speed.update(speed_kmh)

    def on_event(self, label: str, confidence: float,
                 latitude: Optional[float] = None,
                 longitude: Optional[float] = None,
                 speed_kmh: Optional[float] = None) -> tuple[Optional[Alert], str]:
        """
        Handle a confirmed fall / crash detection.

        Returns
        -------
        (alert, reason) – alert is None when suppressed; reason explains why.
        """
        now = self._clock()
        if self.pending is not None:
            return None, "alert_already_pending"
        if now < self._cooldown_until:
            return None, "cooldown"

        max_recent = self.speed.max_recent
        if (self.mode == "car" and self.min_recent_speed_kmh is not None
                and max_recent is not None and max_recent < self.min_recent_speed_kmh):
            logger.info(f"Crash suppressed: vehicle not moving (max recent speed {max_recent:.1f} km/h)")
            return None, "vehicle_not_moving"

        alert = Alert(
            mode=self.mode, label=label, confidence=round(float(confidence), 4),
            created_at=now, deadline=now + self.countdown_s,
            latitude=latitude, longitude=longitude, speed_kmh=speed_kmh,
            max_recent_speed_kmh=max_recent,
        )
        self.pending = alert
        self.history.append(alert)
        logger.warning(f"Alert #{alert.id} PENDING ({label}, conf={confidence:.2f}) – "
                       f"dispatch in {self.countdown_s:.0f}s unless cancelled")
        return alert, "pending"

    def cancel(self, alert_id: Optional[int] = None) -> Optional[Alert]:
        """User confirmed they are OK. Returns the cancelled alert, if any."""
        alert = self.pending
        if alert is None or (alert_id is not None and alert_id != alert.id):
            return None
        alert.state = AlertState.CANCELLED
        self._finish()
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info(f"Alert #{alert.id} CANCELLED by user")
        return alert

    # ------------------------------------------------------------------
    # Countdown (asyncio)
    # ------------------------------------------------------------------

    def start_countdown(self, sink: Optional[EventSink] = None) -> asyncio.Task:
        """Schedule dispatch of the pending alert after the countdown."""
        self._sink = sink
        self._task = asyncio.create_task(self._countdown(self.pending))
        return self._task

    def detach(self) -> None:
        """Client disconnected: stop emitting events, apply disconnect policy."""
        self._sink = None
        if self.pending is not None and not self.dispatch_on_disconnect:
            self.cancel()

    async def _countdown(self, alert: Alert) -> None:
        try:
            await asyncio.sleep(max(0.0, alert.deadline - self._clock()))
        except asyncio.CancelledError:
            return
        if alert.state is not AlertState.PENDING:
            return
        await self.dispatch_now(alert)

    async def dispatch_now(self, alert: Alert) -> bool:
        if self.pending is alert:
            self.pending = None  # point of no return: a late "cancel" must not abort the send
        try:
            ok = await self.dispatcher(alert)
        except Exception as exc:  # never let a notifier failure kill the session
            logger.exception(f"Alert #{alert.id} dispatch error: {exc}")
            ok = False
        alert.state = AlertState.DISPATCHED if ok else AlertState.FAILED
        self._finish()
        logger.critical(f"Alert #{alert.id} {alert.state.value.upper()}")
        await self._emit({"event": f"alert_{alert.state.value}", "alert": alert.to_dict()})
        return ok

    # ------------------------------------------------------------------

    def _finish(self) -> None:
        self.pending = None
        self._cooldown_until = self._clock() + self.cooldown_s

    async def _emit(self, payload: dict) -> None:
        if self._sink is None:
            return
        try:
            await self._sink(payload)
        except Exception:  # socket already closed
            self._sink = None
