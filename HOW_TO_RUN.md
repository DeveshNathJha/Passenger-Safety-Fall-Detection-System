# How to Run & Connect a Phone

## 1. Install

Python 3.10–3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt     # server + training + tests
```

> **Windows tip:** TensorFlow has very long file paths. If `pip` fails with
> `No such file or directory ... .h`, create the venv in a short folder
> (e.g. `C:\venvs\fds`) or enable Windows long-path support.

Just running the server needs only `pip install -r requirements.txt`.

## 2. Start the server

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` lets devices on your Wi-Fi reach it. Open
`http://localhost:8000/docs` for the API, or run it in Docker:

```bash
docker compose up --build
```

## 3. Phone browsers need HTTPS for motion sensors

Android Chrome and iOS Safari only expose accelerometer/gyroscope data on
secure pages. The simplest way on a LAN is [mkcert](https://github.com/FiloSottile/mkcert):

```bash
mkcert -install
mkcert -key-file key.pem -cert-file cert.pem <YOUR_PC_IP> localhost
uvicorn src.app:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

(On Windows, `run_app.bat` detects your IP and uses `key.pem`/`cert.pem` if present.)
Your phone must trust the mkcert root CA, or accept the browser warning once.

Find your PC's IP: `ipconfig` (Windows), `ipconfig getifaddr en0` (macOS), `ip a` (Linux).

## 4. Use the demo page

1. On the phone, open `https://<YOUR_PC_IP>:8000/demo`.
2. iPhone: tap **Allow motion sensors (iOS)**.
3. Pick **Human fall detection** or **Vehicle crash detection** and tap **Connect & stream**.
4. The buffer fills in ~2.6 s; then a prediction arrives every 1.28 s.
5. On a detected fall/crash a full-screen countdown appears. Tap **I'm OK** to
   cancel, otherwise the SOS is sent (logged in mock mode, or a real SMS when
   Twilio is configured in `.env`).

## 5. No phone? Replay a real recording

```bash
# sub7 is a held-out test subject of the deployed model
python scripts/stream_client.py --mobifall <MobiFall>/sub7/FALLS/FOL/FOL_acc_7_1.txt --wait 18
python scripts/stream_client.py --mobifall <MobiFall>/sub7/ADL/JOG/JOG_acc_7_1.txt   # no alert expected
python scripts/stream_client.py --csv data/test_dummy.csv --cancel
```

You'll see the fall window marked as a *candidate*, the post-fall check confirming it
~2.5 s after the impact, `alert_pending`, and `alert_dispatched` 15 s later.

## 6. Retrain from scratch

Datasets (free, manual download):

* **MobiFall v2.0**: Kaggle `kmknation/mobifall-dataset-v20` (login required). Download the
  zip, or put your Kaggle token in `~/.kaggle/kaggle.json` and omit `--local-data`.
* **SisFall**: https://www.mdpi.com/1424-8220/17/1/198 → *Supplementary Materials*
  (the site blocks scripted downloads, so use a browser).

```bash
python src/preprocess.py --task human   --local-data <folder containing MobiFall_Dataset_v2.0>
python src/preprocess.py --task sisfall --local-data <folder containing SisFall_dataset>
python src/preprocess.py --task car          # generates the synthetic dataset

python src/train.py --task human             # deployed model: MobiFall + SisFall, invariant inputs
python src/train.py --task car

# evaluation
python src/train.py --task human --cv 10                                  # 62 subjects, per-dataset breakdown
python src/train.py --task human --data mobifall --loso --features invariant
python src/train.py --task human --data mobifall --compare                # 4 architectures
python src/cross_dataset.py --features raw                                # MobiFall → SisFall
python src/evaluate_events.py --task sisfall --subjects test              # event level, check off vs on
python src/evaluate_events.py --task human --subjects val --grid          # tune the post-fall check
python src/evaluate_events.py --task human --loso                         # event-level LOSO (~25 min)
```

Every command writes its results to `reports/`.
