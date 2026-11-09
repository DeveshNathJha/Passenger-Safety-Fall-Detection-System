"""
src/utils/notifier.py
---------------------
Emergency SOS delivery.

  - TwilioNotifier : real SMS via Twilio (credentials from environment)
  - MockNotifier   : logs the message – used automatically when Twilio
                     is not configured, so the demo works out of the box

Environment variables:
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER,
  EMERGENCY_CONTACT_NUMBER
"""

from __future__ import annotations

import os
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_sos_message(label: str, confidence: float,
                      lat: Optional[float], lon: Optional[float],
                      speed_kmh: Optional[float]) -> str:
    """Compose the SOS text (kept under ~300 chars so it fits in 2 SMS segments)."""
    lines = [
        "EMERGENCY - Passenger Safety System",
        f"{label} (confidence {confidence:.0%}). The user did not cancel the alert.",
    ]
    if speed_kmh is not None:
        lines.append(f"Last speed: ~{speed_kmh:.0f} km/h")
    if lat is not None and lon is not None:
        lines.append(f"Location: https://maps.google.com/?q={lat:.6f},{lon:.6f}")
    else:
        lines.append("Location: GPS unavailable")
    return "\n".join(lines)


class MockNotifier:
    """Logs instead of sending. Always succeeds."""

    name = "mock"

    def __init__(self, to_number: str = ""):
        self.to_number = to_number
        self.sent: list[str] = []

    def send(self, body: str) -> bool:
        self.sent.append(body)
        logger.critical(f"\n{'=' * 50}\n[MOCK SOS] To: {self.to_number or 'UNSET'}\n{body}\n{'=' * 50}")
        return True


class TwilioNotifier:
    """Sends SMS through Twilio. Blocking – call it from a worker thread."""

    name = "twilio"

    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_number: str):
        from twilio.rest import Client  # imported lazily: optional dependency

        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.to_number = to_number

    def send(self, body: str) -> bool:
        try:
            message = self.client.messages.create(body=body, from_=self.from_number, to=self.to_number)
            logger.info(f"Twilio SMS sent. SID: {message.sid}")
            return True
        except Exception as exc:
            logger.error(f"Failed to send Twilio SMS: {exc}")
            return False


def build_notifier():
    """Pick Twilio when fully configured and installed, otherwise fall back to mock."""
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_FROM_NUMBER", "")
    to_number = os.environ.get("EMERGENCY_CONTACT_NUMBER", "")

    if sid and token and from_number and to_number:
        try:
            notifier = TwilioNotifier(sid, token, from_number, to_number)
            logger.info("SOS notifier: Twilio (live SMS)")
            return notifier
        except ImportError:
            logger.warning("Twilio credentials set but `twilio` is not installed – using mock notifier.")
    else:
        logger.info("SOS notifier: mock (set TWILIO_* env vars for live SMS)")
    return MockNotifier(to_number)
