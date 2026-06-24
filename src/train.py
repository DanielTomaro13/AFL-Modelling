"""
Train per-target AFL player-stat models + calibrate dispersion.

Models
  continuous counts (disposals, kicks, handballs, marks, tackles,
    totalClearances, hitouts) + dreamTeamPoints : HistGradientBoostingRegressor
  low counts (goals, behinds)                   : HGBR with Poisson loss -> rate λ

Validation
  season holdout (default last 2 seasons): MAE vs trailing-5 baseline.

Outputs
  models/afl_models.joblib        fitted models + feature cols + meta
  artifacts/dispersion.json       per-target residual spread (for distribution pricing)
  reports/holdout_metrics.csv     accuracy table (the trust numbers)
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

FEAT = "data/processed/features.parquet"
COLS = "artifacts/feature_cols.json"
MODELS = "models/afl_models.joblib"
DISP = "artifacts/dispersion.json"
METRICS = "reports/holdout_metrics.csv"

CONT = ["disposals", "kicks", "handballs", "marks", "tackles",
        "totalClearances", "hitouts", "dreamTeamPoints"]
POIS = ["goals", "behinds"]
TARGETS = CONT + POIS
HOLDOUT = [int(s) for s in os.environ.get("AFL_HOLDOUT", "2024,2025").split(",")]


def make_model(poisson=False):
    return HistGradientBoostingRegressor(
        loss="poisson" if poisson else "squared_error",
        learning_rate=0.06, max_iter=450, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        random_state=7,
    )


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    df = pd.read_parquet(FEAT)
    feat = json.load(open(COLS))
    feat = [c for c in feat if c in df.columns]

    tr = df[~df["season"].isin(HOLDOUT)]
    te = df[df["season"].isin(HOLDOUT)]
    print(f"train {len(tr):,} rows (<{min(HOLDOUT)})  holdout {len(te):,} rows {HOLDOUT}")

    models, disp, rows = {}, {}, []
    for t in TARGETS:
        poisson = t in POIS
        m = make_model(poisson=poisson)
        mtr = tr[tr[t].notna()]
        m.fit(mtr[feat], mtr[t].clip(lower=0))
        models[t] = m

        mte = te[te[t].notna()].copy()
        pred = np.clip(m.predict(mte[feat]), 0, None)
        base_col = f"{t}_r5"
        base = mte[base_col].fillna(mtr[t].mean()) if base_col in mte else None
        model_mae = mae(mte[t], pred)
        base_mae = mae(mte[t], base) if base is not None else np.nan
        impr = (base_mae - model_mae) / base_mae * 100 if base_mae else np.nan

        resid = mte[t].to_numpy() - pred
        resid_std = float(np.std(resid))
        mean_pred = float(np.mean(pred)) or 1.0
        disp[t] = {
            "type": "poisson" if poisson else "normal",
            "resid_std": round(resid_std, 4),
            "disp_pct": round(resid_std / mean_pred, 4),   # dispersion as CoV (resid_std / mean)
            "mean": round(float(mte[t].mean()), 4),
        }
        rows.append({"target": t, "type": disp[t]["type"], "n": len(mte),
                     "model_mae": round(model_mae, 3),
                     "baseline_mae": round(base_mae, 3),
                     "improvement_pct": round(impr, 1)})
        print(f"  {t:16s} MAE {model_mae:6.3f}  base {base_mae:6.3f}  "
              f"{impr:+5.1f}%  σ={resid_std:.2f}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump({"models": models, "features": feat, "targets": TARGETS,
                 "holdout": HOLDOUT}, MODELS)
    json.dump(disp, open(DISP, "w"), indent=2)
    pd.DataFrame(rows).to_csv(METRICS, index=False)
    # trust artifact for the site's Backtest page
    json.dump({
        "holdout_seasons": HOLDOUT,
        "n_train": int(len(tr)), "n_holdout": int(len(te)),
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "metrics": rows,
    }, open("artifacts/backtest.json", "w"), indent=2)
    beat = sum(1 for r in rows if r["improvement_pct"] > 0)
    print(f"\nWrote {MODELS}, {DISP}, {METRICS}. "
          f"{beat}/{len(rows)} targets beat trailing-5 baseline.")


if __name__ == "__main__":
    main()
