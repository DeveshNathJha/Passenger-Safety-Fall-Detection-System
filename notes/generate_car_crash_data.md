# Car Crash Data Generator (`scripts/generate_car_crash_data.py`)

## Purpose
Because 6-axis (Accelerometer + Gyroscope) datasets sampled at 50 Hz for in-vehicle passenger accidents are relatively scarce in the open-source community, this script synthesizes a realistic substitute to train the vehicle AI model.

## Mechanics
*   **Normal Driving State**: Simulates regular road bumps, minor braking, and turning with variance constraints between 0.5g and 2.0g.
*   **Collision State**: Synthesizes abrupt deceleration spikes (10g - 20g shocks on the Y/Z axes) coupled with intense rotational velocity (gyroscope spikes) signifying a tumbling or violently stopping vehicle.
*   **Output**: Generates formatted CSVs matching the strict `(N, 128, 6)` sliding window constraints required by the system, saved into `data/raw/synthetic_car_crash/`.
