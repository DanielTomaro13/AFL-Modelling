# AFL Player-Stat Modelling

Predicts ten per-player, per-match markets for AFL men's matches and feeds them
to the [AFL-23-0](https://github.com/DanielTomaro13/AFL-23-0) site, where a
TypeScript Monte-Carlo engine turns them into full distributions, value edges
and Kelly staking.

**Markets:** Disposals · Goals · Kicks · Handballs · Marks · Tackles · Behinds ·
Clearances · Hit Outs · AFL Fantasy (DreamTeam) points.

## Architecture

```
Champion Data StatsPro (api.afl.com.au)
        │  src/ingest.py   (anon WMCTok; sweep seasons × rounds)
        ▼
data/processed/player_match.parquet      one row per player per match (2015–now)
        │  src/features.py  (leakage-safe rollups, role, opponent-conceded)
        ▼
data/processed/features.parquet
        │  src/train.py     (GBM per count stat + Poisson goals/behinds)
        ▼
models/afl_models.joblib · artifacts/dispersion.json · reports/holdout_metrics.csv
        │  src/predict.py   (one round → per-player expecteds + shares + dispersion)
        ▼
artifacts/projection_inputs.json   ──►  AFL-23-0 pipeline/src/montecarlo (TS)
```

The Python side is the "brain" (training, calibration); the TS Monte-Carlo in
AFL-23-0 is the "engine" that produces the joint, correlated per-player
distributions the site renders.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Ingest | `python src/ingest.py` | `data/processed/player_match.parquet` (~107k rows, 2015–2026) |
| Features | `python src/features.py` | `data/processed/features.parquet` (180 leakage-safe features) |
| Train | `python src/train.py` | models + `artifacts/dispersion.json` + `reports/holdout_metrics.csv` |
| Predict a round | `python src/predict.py [season round]` | `artifacts/projection_inputs.json` |

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python src/ingest.py && python src/features.py && python src/train.py
python src/predict.py                 # auto: next round; or e.g. `... 2026 16`
```

Env: `AFL_SEASONS=2018,2019,...` to limit ingest; `AFL_HOLDOUT=2024,2025` to set
the validation seasons.

## Models & validation

- **Counts** (disposals, kicks, handballs, marks, tackles, clearances, hit-outs,
  fantasy): `HistGradientBoostingRegressor`.
- **Low counts** (goals, behinds): Poisson-loss GBM → rate λ for anytime / 2+ / 3+.
- **Leakage safety:** every trailing feature is shifted one game; a row never
  sees its own result. Validation is a **season holdout** (default 2024–25).
- Out of sample the models beat a trailing-5-game baseline on **9 of 10** markets
  (disposals +5.9%, tackles +6.3%, fantasy +5.8% MAE). Hit-outs are ruck-locked
  and roughly match the baseline — see `reports/holdout_metrics.csv`.
- **Fantasy** is modelled twice — directly, and rebuilt from components in the
  Monte-Carlo — as a consistency cross-check.

## Data source

Public AFL StatsPro feed (Champion Data). `src/afl_api.py` mints the anonymous
`x-media-mis-token` and reads `playersStats/rounds/{id}` (bulk) and
`matchRosters/round/{id}` (fixtures). No API key. Every fetch is cached to
`data/raw/` (git-ignored) so re-runs are free and resumable.
