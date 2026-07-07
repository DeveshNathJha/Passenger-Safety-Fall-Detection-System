# Notifier Utility (`src/utils/notifier.py`)

## Purpose
This utility is responsible for sending out emergency SOS notifications when a vehicle crash is detected. It serves as the primary external communication hook for the passenger safety system.

## Key Features
* **Twilio Integration**: Uses the Twilio Python SDK to dispatch SMS alerts.
* **Graceful Degradation**: If the Twilio API credentials (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, etc.) are missing from the environment, the system automatically falls back to a "Mock Mode", printing the critical alert strictly to the system logger.
* **Sensor Fusion Payload**: Constructs an SMS containing the exact dynamic speed at the time of impact, alongside a Google Maps URL pointing to the user's `latitude` and `longitude`.

## Usage in App
Called seamlessly from the `websocket_predict` loop in `src/app.py` when the car crash model yields a high-confidence fall/crash alongside an abrupt drop in GPS velocity.
