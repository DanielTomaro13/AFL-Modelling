"""
Train per-target AFL player-stat models + calibrate dispersion.

Models
  continuous counts (disposals, kicks, handballs, marks, tackles,
    totalClearances, hitouts) + dreamTeamPoints : HistGradientBoostingRegressor
  low counts (goals, behinds)                   : HGBR with Poisson loss -> rate λ

Validation, then refit
  1. fit on everything OUTSIDE the holdout seasons (default last 2), score MAE
     vs trailing-5 baseline on the holdout — these are the trust numbers;
  2. calibrate dispersion from the holdout residuals:
       sigma(pred) = sigma_a + sigma_b·sqrt(pred)   per-player spread (normal targets)
       phi = Var(resid)/mean                        overdispersion (poisson targets)
       team_cv = std of (actual − expected)/expected team totals  (for the MC)
  3. REFIT on ALL seasons with the same hyperparameters — the shipped model
     must not exclude the most recent completed seasons.
  Training uses recency weights (halflife 3 seasons).

Outputs
  models/afl_models.joblib        production models (refit on all data)
  models/afl_models_val.joblib    validation models (holdout-blind; for review.py)
  artifacts/dispersion.json       per-target dispersion + sigma model + team_cv
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
MODELS_VAL = "models/afl_models_val.joblib"
DISP = "artifacts/dispersion.json"
METRICS = "reports/holdout_metrics.csv"

CONT = ["disposals", "kicks", "handballs", "marks", "tackles",
        "totalClearances", "hitouts", "dreamTeamPoints"]
POIS = ["goals", "behinds"]
TARGETS = CONT + POIS
HOLDOUT = [int(s) for s in os.environ.get("AFL_HOLDOUT", "2024,2025").split(",")]
WEIGHT_HALFLIFE = 3.0  # seasons


def make_model(poisson=False):
    return HistGradientBoostingRegressor(
        loss="poisson" if poisson else "squared_error",
        learning_rate=0.06, max_iter=450, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1,
        random_state=7,
    )


def recency_weights(seasons):
    """Exponential decay by season age; halflife WEIGHT_HALFLIFE seasons."""
    age = seasons.max() - seasons
    return np.power(0.5, age / WEIGHT_HALFLIFE)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def fit_sigma(pred, resid, bins=12):
    """Heteroscedastic residual spread: sigma(pred) ≈ a + b·sqrt(pred).
    Weighted LSQ over prediction-quantile bins. Returns (a, b, floor)."""
    d = pd.DataFrame({"p": pred, "r": resid})
    q = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby(q, observed=True)
    mp, sd, n = g["p"].mean(), g["r"].std(), g["r"].size()
    ok = sd.notna() & (n >= 20)
    mp, sd, n = mp[ok], sd[ok], n[ok]
    X = np.column_stack([np.ones(len(mp)), np.sqrt(mp)])
    w = np.sqrt(n.to_numpy(float))
    coef, *_ = np.linalg.lstsq(X * w[:, None], sd.to_numpy() * w, rcond=None)
    floor = float(max(0.2, sd.min() * 0.8))
    return float(coef[0]), float(coef[1]), floor


def main():
    df = pd.read_parquet(FEAT)
    feat = json.load(open(COLS))
    feat = [c for c in feat if c in df.columns]

    tr = df[~df["season"].isin(HOLDOUT)]
    te = df[df["season"].isin(HOLDOUT)]
    print(f"train {len(tr):,} rows  holdout {len(te):,} rows {HOLDOUT}")

    val_models, models, disp, rows = {}, {}, {}, []
    for t in TARGETS:
        poisson = t in POIS

        # -- 1. validation fit (holdout-blind) --------------------------------
        mtr = tr[tr[t].notna()]
        mv = make_model(poisson=poisson)
        mv.fit(mtr[feat], mtr[t].clip(lower=0),
               sample_weight=recency_weights(mtr["season"]))
        val_models[t] = mv

        mte = te[te[t].notna()].copy()
        pred = np.clip(mv.predict(mte[feat]), 0, None)
        base = mte[f"{t}_r5"].fillna(mtr[t].mean())
        model_mae = mae(mte[t], pred)
        base_mae = mae(mte[t], base)
        impr = (base_mae - model_mae) / base_mae * 100 if base_mae else np.nan

        # -- 2. dispersion calibration from holdout residuals -----------------
        y = mte[t].to_numpy(float)
        resid = y - pred
        resid_std = float(np.std(resid))
        mean_pred = float(np.mean(pred)) or 1.0
        sig_a, sig_b, sig_floor = fit_sigma(pred, resid)
        # team-level CV: spread of actual vs model-expected TEAM totals — this
        # is the number the TS Monte-Carlo should draw team totals with
        gt = (mte.assign(_p=pred).groupby(["match_id", "team_id"])
              .agg(exp=("_p", "sum"), act=(t, "sum")))
        gt = gt[gt["exp"] > 0]
        team_cv = float(((gt["act"] - gt["exp"]) / gt["exp"]).std())
        disp[t] = {
            "type": "poisson" if poisson else "normal",
            "resid_std": round(resid_std, 4),
            "disp_pct": round(resid_std / mean_pred, 4),
            "mean": round(float(mte[t].mean()), 4),
            "sigma_a": round(sig_a, 4), "sigma_b": round(sig_b, 4),
            "sigma_floor": round(sig_floor, 4),
            "phi": round(float(np.var(resid) / max(mean_pred, 1e-6)), 4),
            "team_cv": round(team_cv, 4),
        }
        rows.append({"target": t, "type": disp[t]["type"], "n": len(mte),
                     "model_mae": round(model_mae, 3),
                     "baseline_mae": round(base_mae, 3),
                     "improvement_pct": round(impr, 1),
                     "team_cv": round(team_cv, 3)})
        print(f"  {t:16s} MAE {model_mae:6.3f}  base {base_mae:6.3f}  "
              f"{impr:+5.1f}%  σ={resid_std:.2f}  team_cv={team_cv:.3f}")

        # -- 3. production refit on ALL data ----------------------------------
        ma = df[df[t].notna()]
        mf = make_model(poisson=poisson)
        mf.fit(ma[feat], ma[t].clip(lower=0),
               sample_weight=recency_weights(ma["season"]))
        models[t] = mf

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump({"models": models, "features": feat, "targets": TARGETS,
                 "holdout": HOLDOUT, "refit_all_data": True}, MODELS)
    joblib.dump({"models": val_models, "features": feat, "targets": TARGETS,
                 "holdout": HOLDOUT}, MODELS_VAL)
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
    print(f"\nWrote {MODELS} (refit on all seasons), {MODELS_VAL}, {DISP}, {METRICS}. "
          f"{beat}/{len(rows)} targets beat trailing-5 baseline.")


if __name__ == "__main__":
    main()
