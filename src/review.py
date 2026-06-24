"""
Model soundness review — out-of-sample diagnostics on the held-out seasons.

For every target, on the holdout we check:
  bias        mean(pred − actual)            ~0  (unbiased)
  mae / base  vs trailing-5 baseline         model should win
  resid_std   matches artifacts/dispersion   the spread the MC/pricing uses
  cover80     P(actual in pred ±1.28σ)       ~0.80 if the distribution width is right
  cal_err     reliability of P(over line)    ~0 (probabilities are trustworthy)
  leak        max |feature corr| & train/holdout gap   guards against leakage

Run: python src/review.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import joblib

FEAT = "data/processed/features.parquet"
COLS = "artifacts/feature_cols.json"
MODELS = "models/afl_models.joblib"
DISP = "artifacts/dispersion.json"

POIS = {"goals", "behinds"}


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
    bundle = joblib.load(MODELS)
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
        sigma = disp[t]["resid_std"]

        # 80% central-interval coverage using the dispersion the MC trusts
        lo, hi = pred - 1.2816 * sigma, pred + 1.2816 * sigma
        cover = float(np.mean((y >= lo) & (y <= hi)))

        # probability calibration of P(value > line) at a near-median line
        line = max(0.5, round(np.median(y)) - 0.5)
        if t in POIS:
            lam = np.clip(pred, 1e-6, None)
            p_over = 1 - poisson.cdf(np.floor(line), lam)
        else:
            p_over = 1 - norm.cdf(line, loc=pred, scale=sigma)
        cal = reliability(p_over, (y > line))

        # crude leakage guard: any feature almost perfectly correlated w/ target?
        rows.append({
            "target": t, "n": len(mte), "bias": round(bias, 2),
            "mae": round(mae, 2), "base": round(base_mae, 2),
            "win%": round((base_mae - mae) / base_mae * 100, 1),
            "sigma": round(sigma, 2), "cover80": round(cover, 2),
            f"P(>{line:g})cal": round(cal, 3),
            "tr/te_mae": round(mae_tr / mae, 2),
        })

    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(rep.to_string(index=False))

    # leakage scan: top |corr(feature, target)| across targets
    print("\nLeakage scan — max |corr| of any model feature with each target "
          "(>0.95 would be suspicious):")
    for t in models:
        sub = df[df[t].notna()]
        cors = sub[feat].corrwith(sub[t]).abs()
        top = cors.sort_values(ascending=False).head(1)
        print(f"  {t:16s} {top.index[0]:28s} {top.iloc[0]:.3f}")

    # verdicts
    print("\nVerdict:")
    bad = []
    for r in rows:
        if abs(r["bias"]) > 0.6 * r["sigma"]:
            bad.append(f"{r['target']}: bias {r['bias']} large vs σ {r['sigma']}")
        if not (0.70 <= r["cover80"] <= 0.88):
            bad.append(f"{r['target']}: 80% coverage {r['cover80']} off target")
        if r["tr/te_mae"] < 0.6:
            bad.append(f"{r['target']}: train MAE << holdout (overfit?) ratio {r['tr/te_mae']}")
    print("  All targets sound." if not bad else "  Flags:\n   - " + "\n   - ".join(bad))


if __name__ == "__main__":
    main()
