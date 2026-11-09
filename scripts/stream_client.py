"""
scripts/stream_client.py
------------------------
Replay a recording through the LIVE server over WebSocket, exactly like
the phone does (timestamped samples, 100 ms batches) – an end-to-end test
of resampling → windowing → model → alert workflow without a phone.

Sources:
    --mobifall FILE   a MobiFall *_acc_* file (the matching *_gyro_* is used too);
                      raw native-rate timestamps are sent, the server resamples
    --csv FILE        50 Hz CSV acc_x..gyro_z (e.g. data/test_dummy.csv)

Usage:
    uvicorn src.app:app --port 8000 &
    python scripts/stream_client.py --mobifall <MobiFall>/sub1/FALLS/FOL/FOL_acc_1_1.txt
    python scripts/stream_client.py --csv data/test_dummy.csv --cancel
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import parse_sensor_file

KEYS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def mobifall_samples(acc_path: str) -> list[dict]:
    """Interleave raw acc and gyro events in time order (as a phone delivers them)."""
    acc = parse_sensor_file(acc_path)
    gyro = parse_sensor_file(acc_path.replace("_acc_", "_gyro_"))
    t0 = max(acc[0, 0], gyro[0, 0])
    t1 = min(acc[-1, 0], gyro[-1, 0])
    out, g = [], gyro
    for t, ax, ay, az in acc:
        if not t0 <= t <= t1:
            continue
        gx, gy, gz = np.rad2deg([np.interp(t, g[:, 0], g[:, c]) for c in (1, 2, 3)])
        out.append(dict(zip(KEYS, map(float, (ax, ay, az, gx, gy, gz))), t=(t - t0) * 1000.0))
    return out


def csv_samples(path: str, hz: float = 50.0) -> list[dict]:
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    return [dict(zip(KEYS, map(float, r[:6])), t=i * 1000.0 / hz) for i, r in enumerate(rows)]


async def run(url: str, samples: list[dict], cancel: bool, realtime: bool, wait_s: float) -> None:
    import websockets

    async with websockets.connect(url, max_size=2**20) as ws:
        print(json.loads(await ws.recv()))

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                ev = msg["event"]
                if ev == "prediction":
                    flag = ("<<< EVENT" if msg["is_event"]
                            else "<<< candidate, checking stillness" if msg.get("candidate") else "")
                    print(f"window {msg['total_inferences']:>3}  score {msg['raw_score']:.3f}  "
                          f"(thr {msg['threshold']:.3f})  {msg['latency_ms']:.2f} ms  {flag}")
                elif ev == "confirmation":
                    verdict = "CONFIRMED (person still)" if msg["is_event"] else "rejected (still moving)"
                    print(f"    post-fall check: {verdict}  stillness={msg.get('stillness')}  "
                          f"decided {msg.get('delay_s')} s after impact")
                elif ev.startswith("alert"):
                    print(f"*** {ev}: {msg.get('alert', msg.get('reason'))}")
                    if ev == "alert_pending" and cancel:
                        await ws.send(json.dumps({"type": "cancel_alert", "alert_id": msg["alert"]["id"]}))
                elif ev == "error":
                    print("error:", msg["detail"])

        task = asyncio.create_task(reader())
        batch, last_flush = [], samples[0]["t"]
        for s in samples:
            batch.append(s)
            if s["t"] - last_flush >= 100:
                await ws.send(json.dumps({"type": "batch", "samples": batch}))
                batch, last_flush = [], s["t"]
                if realtime:
                    await asyncio.sleep(0.1)
        if batch:
            await ws.send(json.dumps({"type": "batch", "samples": batch}))
        await asyncio.sleep(wait_s)  # let a pending alert count down
        task.cancel()


def main():
    p = argparse.ArgumentParser(description="Stream a recording to the live server.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mobifall", help="path to a MobiFall *_acc_* file")
    src.add_argument("--csv", help="50 Hz CSV acc_x..gyro_z")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/predict")
    p.add_argument("--mode", default="human")
    p.add_argument("--api-key", default=None)
    p.add_argument("--cancel", action="store_true", help="tap 'I'm OK' as soon as an alert is pending")
    p.add_argument("--realtime", action="store_true", help="send at wall-clock speed")
    p.add_argument("--wait", type=float, default=2.0, help="seconds to keep listening at the end")
    a = p.parse_args()

    samples = mobifall_samples(a.mobifall) if a.mobifall else csv_samples(a.csv)
    url = f"{a.url}?model={a.mode}" + (f"&api_key={a.api_key}" if a.api_key else "")
    print(f"Streaming {len(samples)} samples to {url}")
    asyncio.run(run(url, samples, a.cancel, a.realtime, a.wait))


if __name__ == "__main__":
    main()
