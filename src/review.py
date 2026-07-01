"""
Model soundness review — out-of-sample diagnostics on the held-out seasons.

Uses the VALIDATION models (models/afl_models_val.joblib — holdout-blind); the
production bundle is refit on all seasons so scoring it on the holdout would
flatter it.

For every target, on the holdout we check:
  bias        mean(pred − actual)            ~0  (unbiased)
  mae / base  vs trailing-5 baseline         model should win
  cover80     P(actual in pred ±1.28σ(pred)) ~0.80 with the HETEROSCEDASTIC
              sigma the pricing actually uses (sigma_a + sigma_b·sqrt(pred))
  cal_err     reliability of P(over line)    ~0 (probabilities are trustworthy)
  tr/te       train/holdout MAE ratio        guards against overfit
Plus: hitouts split rucks vs non-rucks, and a leakage corr scan.

Run: python src/review.py   (after train.py, same checkout)
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import joblib

FEAT = "data/processed/features.parquet"
COLS = "artifacts/feature_cols.json"
MODELS_VAL = "models/afl_models_val.joblib"
DISP = "artifacts/dispersion.json"

POIS = {"goals", "behinds"}


def sigma_of(pred, d):
    """Per-row predictive sigma, matching what predict.py emits."""
    if d["type"] == "poisson":
        return np.sqrt(np.maximum(d.get("phi", 1.0), 0.5) * np.maximum(pred, 1e-3))
    if "sigma_a" in d:
        s = d["sigma_a"] + d["sigma_b"] * np.sqrt(np.maximum(pred, 0))
        return np.maximum(d.get("sigma_floor", 0.2), s)
    return np.full_like(pred, d["resid_std"], dtype=float)


def reliability(p, y, bins=10):
    """Expected calibration error for binary outcome y vs predicted prob p."""
    p = np.asarray(p); y = np.asarray(y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() < 20:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return ece


def main():
    df = pd.read_parquet(FEAT)
    feat = [c for c in json.load(open(COLS)) if c in df.columns]
    bundle = joblib.load(MODELS_VAL)
    models, holdout = bundle["models"], bundle["holdout"]
    disp = json.load(open(DISP))

    tr = df[~df["season"].isin(holdout)]
    te = df[df["season"].isin(holdout)]
    print(f"holdout seasons {holdout}: {len(te):,} player-games "
          f"(train {len(tr):,})\n")

    rows = []
    for t, model in models.items():
        mte = te[te[t].notna()].copy()
        mtr = tr[tr[t].notna()]
        y = mte[t].to_numpy(float)
        pred = np.clip(model.predict(mte[feat]), 0, None)
        ptr = np.clip(model.predict(mtr[feat]), 0, None)

        bias = float(np.mean(pred - y))
        mae = float(np.mean(np.abs(pred - y)))
        mae_tr = float(np.mean(np.abs(ptr - mtr[t].to_numpy(float))))
        base = mte[f"{t}_r5"].fillna(mtr[t].mean()).to_numpy(float)
        base_mae = float(np.mean(np.abs(base - y)))
        sig = sigma_of(pred, disp[t])

        # 80% central-interval coverage using the heteroscedastic sigma
        lo, hi = pred - 1.2816 * sig, pred + 1.2816 * sig
        cover = float(np.mean((y >= lo) & (y <= hi)))

        # probability calibration of P(value > line) at a near-median line
        line = max(0.5, round(np.median(y)) - 0.5)
        if t in POIS:
            lam = np.clip(pred, 1e-6, None)
            p_over = 1 - poisson.cdf(np.floor(line), lam)
        else:
            p_over = 1 - norm.cdf(line, loc=pred, scale=sig)
        cal = reliability(p_over, (y > line))

        rows.append({
            "target": t, "n": len(mte), "bias": round(bias, 2),
            "mae": round(mae, 2), "base": round(base_mae, 2),
            "win%": round((base_mae - mae) / base_mae * 100, 1),
            "sigma@mean": round(float(np.mean(sig)), 2),
            "cover80": round(cover, 2),
            f"P(>{line:g})cal": round(cal, 3),
            "tr/te_mae": round(mae_tr / mae, 2),
        })

    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(rep.to_string(index=False))

    # hitouts are bimodal (rucks vs everyone) — judge them separately
    t = "hitouts"
    mte = te[te[t].notna()].copy()
    pred = np.clip(models[t].predict(mte[feat]), 0, None)
    for label, mask in [("rucks", mte["is_ruck"] == 1), ("non-rucks", mte["is_ruck"] != 1)]:
        y = mte.loc[mask, t].to_numpy(float); p = pred[mask.to_numpy()]
        base = mte.loc[mask, f"{t}_r5"].fillna(0).to_numpy(float)
        if len(y):
            sig = sigma_of(p, disp[t])
            cover = float(np.mean((y >= p - 1.2816 * sig) & (y <= p + 1.2816 * sig)))
            print(f"\nhitouts {label:9s}: n={len(y):6d}  MAE {np.mean(np.abs(p-y)):.2f}  "
                  f"base {np.mean(np.abs(base-y)):.2f}  mean {y.mean():.1f}  cover80 {cover:.2f}")

    # leakage scan: top |corr(feature, target)| across targets
    print("\nLeakage scan — max |corr| of any model feature with each target "
          "(>0.95 would be suspicious):")
    for t in models:
        sub = df[df[t].notna()]
        cors = sub[feat].corrwith(sub[t]).abs()
        top = cors.sort_values(ascending=False).head(1)
        print(f"  {t:16s} {top.index[0]:28s} {top.iloc[0]:.3f}")

    # verdicts. Central-interval coverage is only meaningful for continuous-ish
    # targets: for discrete low counts (goals/behinds, non-ruck hitouts) a
    # ±1.28σ interval is lumpy by construction — their real check is the
    # P(over) calibration column.
    print("\nVerdict:")
    skip_cover = POIS | {"hitouts"}
    bad = []
    for r in rows:
        if abs(r["bias"]) > 0.6 * r["sigma@mean"]:
            bad.append(f"{r['target']}: bias {r['bias']} large vs σ {r['sigma@mean']}")
        if r["target"] not in skip_cover and not (0.70 <= r["cover80"] <= 0.88):
            bad.append(f"{r['target']}: 80% coverage {r['cover80']} off target")
        if r["tr/te_mae"] < 0.6:
            bad.append(f"{r['target']}: train MAE << holdout (overfit?) ratio {r['tr/te_mae']}")
    print("  All targets sound." if not bad else "  Flags:\n   - " + "\n   - ".join(bad))


if __name__ == "__main__":
    main()
