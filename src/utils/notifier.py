"""
src/utils/notifier.py
---------------------
Handles external notifications (SMS, Email) for emergency SOS alerts.
Currently integrates with Twilio API to send SMS upon car crash detection.
"""
import os
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import Twilio SDK; if missing, fail gracefully.
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


class EmergencyNotifier:
    """
    Sends emergency alerts using Twilio.
    Expects environment variables or config keys:
      - TWILIO_ACCOUNT_SID
      - TWILIO_AUTH_TOKEN
      - TWILIO_FROM_NUMBER
      - EMERGENCY_CONTACT_NUMBER
    """
    def __init__(self):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER", "")
        self.to_number = os.environ.get("EMERGENCY_CONTACT_NUMBER", "")
        
        self.is_configured = bool(self.account_sid and self.auth_token and self.from_number and self.to_number)
        
        if self.is_configured and TWILIO_AVAILABLE:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            self.client = None

    def send_sos_sms(self, lat: Optional[float], lon: Optional[float], speed_kmh: Optional[float]) -> bool:
        """
        Constructs and sends an SOS SMS containing Google Maps location and last known speed.
        If Twilio is not configured, logs a mock SMS instead.
        """
        # Construct message
        msg_body = " EMERGENCY \nVehicle crash detected by Passenger Safety System!\n\n"
        
        if speed_kmh is not None:
            msg_body += f"Impact Speed: ~{speed_kmh:.1f} km/h\n"
            
        if lat is not None and lon is not None:
            msg_body += f"Location: https://maps.google.com/?q={lat},{lon}"
        else:
            msg_body += "Location: GPS data unavailable."

        # Mock Mode (No credentials)
        if not self.is_configured or not TWILIO_AVAILABLE:
            logger.critical(f"\n{'='*50}\n[MOCK SOS ALERT TRIGGERED]\nTo: {self.to_number or 'UNKNOWN'}\n{msg_body}\n{'='*50}")
            logger.warning("Twilio credentials or SDK not found. Skipping live SMS.")
            return True

        # Live Mode
        try:
            logger.info("Attempting to send live Twilio SOS SMS...")
            message = self.client.messages.create(
                body=msg_body,
                from_=self.from_number,
                to=self.to_number
            )
            logger.info(f"Twilio SMS sent successfully! SID: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Twilio SMS: {e}")
            return False

# Global singleton
notifier = EmergencyNotifier()
