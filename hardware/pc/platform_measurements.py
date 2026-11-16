"""Platform values reported in Table 5 (measured in this study), from the identification logs.
Usage: python platform_measurements.py --logs ../data/ident_logs --out ../results
- control loop: longest period between logged control steps (all runs)
- speed measurement: median interval between changes of the measured wheel speed
- odometry: integrated distance of the speed checks versus the taped distance (taped.csv)
- start from rest: time from the first nonzero speed command to the first measured speed > 0.01 m/s,
  for the step-command runs (speed checks, acceleration step) and the ramped starts (braking runs)."""
import argparse, csv, glob, json, os
import numpy as np

def load(f):
    r = list(csv.reader(open(f))); h = r[1]
    rows = [dict(zip(h, x)) for x in r[2:]]
    col = lambda k: np.array([float(x[k]) for x in rows])
    return r[0], col

ap = argparse.ArgumentParser(); ap.add_argument("--logs", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.logs, "ident_*.csv")))
loop, upd, dead = [], [], {}
for f in files:
    meta, c = load(f); t, v, vc = c("t"), c("v_meas"), c("v_cmd")
    loop.append(float(np.diff(t).max()))
    ch = t[1:][np.diff(v) != 0]
    if len(ch) > 3: upd.append(float(np.median(np.diff(ch))))
    rid = meta[1]
    i0, i1 = int(np.argmax(vc > 0.01)), int(np.argmax(v > 0.01))
    dead[rid] = round(float(t[i1] - t[i0]), 3)
taped = {r["run_id"]: float(r["taped_m"]) for r in csv.DictReader(open(os.path.join(a.logs, "taped.csv")))}
odo = {}
for rid, d in taped.items():
    meta, c = load(os.path.join(a.logs, "ident_%s.csv" % rid))
    x = float(c("x_int")[-1]); odo[rid] = {"integrated_m": round(x, 4), "taped_m": d, "error_pct": round(100 * (x - d) / d, 2)}
step = [dead[k] for k in dead if k.startswith(("SPD", "ACC"))]
ramped = [dead[k] for k in dead if k.startswith(("BRK-", "BRKI"))]
res = {"n_runs": len(files),
       "longest_loop_period_ms": round(1000 * max(loop), 1),
       "speed_update_interval_ms": round(1000 * float(np.median(upd)), 1),
       "odometry": odo,
       "odometry_error_pct_range": [min(o["error_pct"] for o in odo.values()), max(o["error_pct"] for o in odo.values())],
       "start_delay_s": dead,
       "start_delay_step_runs_s": [min(step), max(step)],
       "start_delay_ramped_runs_s": [min(ramped), max(ramped)],
       "note": "random-input runs excluded from the start-delay ranges: their first command is random in size"}
json.dump(res, open(os.path.join(a.out, "platform_measurements.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("odometry", "start_delay_s")}, indent=1))
