"""
Produce per-player projection inputs for one round -> artifacts/projection_inputs.json

This is the bridge the TS Monte-Carlo engine (in AFL-23-0) consumes. For each
fixture it emits, per player: expected value of every target, the player's share
of the team total, and the dispersion% — all derived empirically from the models.

Lineups: confirmed roster if available, else a proxy = each team's most recent XVIII.

Usage:
  python src/predict.py <season> <round>        # e.g. 2026 15
  python src/predict.py                          # auto: latest round with a fixture
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd

import afl_api as api
import features as F

HIST = "data/processed/player_match.parquet"
MODELS = "models/afl_models.joblib"
DISP = "artifacts/dispersion.json"
OUT = "artifacts/projection_inputs.json"

TARGETS = F.TARGETS


def fixtures_for(season, rnd):
    rosters = api.match_rosters_round(season, rnd)
    out = []
    for item in (rosters or []):
        mt = (item or {}).get("match") or {}
        vn = (item or {}).get("venue") or {}
        if not mt.get("homeTeamId"):
            continue
        out.append({
            "match_id": mt.get("matchId"),
            "home_team_id": mt.get("homeTeamId"),
            "away_team_id": mt.get("awayTeamId"),
            "venue": vn.get("name"), "venue_state": vn.get("state"),
            "date": mt.get("utcStartTime") or mt.get("date"),
            "status": mt.get("status"),
        })
    return out


def proxy_lineup(hist, team_id, before_date):
    """Players from this team's most recent match strictly before `before_date`."""
    h = hist[(hist["team_id"] == team_id) & (hist["date"] < before_date)]
    if h.empty:
        return h
    last = h["date"].max()
    return h[h["date"] == last]


def build_placeholders(hist, fixtures, season, rnd):
    rows = []
    for fx in fixtures:
        bd = pd.to_datetime(fx["date"], utc=True) if fx["date"] else hist["date"].max()
        for side, tid, oid in [("home", fx["home_team_id"], fx["away_team_id"]),
                               ("away", fx["away_team_id"], fx["home_team_id"])]:
            lu = proxy_lineup(hist, tid, bd)
            for _, p in lu.iterrows():
                rows.append({
                    "season": season, "round": rnd, "round_id": api.round_provider(season, rnd),
                    "match_id": fx["match_id"], "date": bd,
                    "venue": fx["venue"], "venue_state": fx["venue_state"],
                    "temp_c": np.nan, "weather": None,
                    "is_home": 1 if side == "home" else 0,
                    "player_id": p["player_id"], "player": p["player"],
                    "given_name": p["given_name"], "surname": p["surname"],
                    "position": p["position"], "kicking_foot": p["kicking_foot"],
                    "age": p["age"], "height_cm": p["height_cm"], "jumper": p["jumper"],
                    "team_id": tid, "team": p["team"], "team_abbr": p["team_abbr"],
                    "opponent_id": oid, "opponent": None, "opponent_abbr": None,
                    "outcome": None, "team_score": np.nan, "opp_score": np.nan, "margin": np.nan,
                })
    # align to history columns; stat columns not set here become NaN placeholders
    return pd.DataFrame(rows).reindex(columns=hist.columns)


def main():
    import joblib
    bundle = joblib.load(MODELS)
    models, feat = bundle["models"], bundle["features"]
    disp = json.load(open(DISP))
    hist = pd.read_parquet(HIST)

    if len(sys.argv) >= 3:
        season, rnd = int(sys.argv[1]), int(sys.argv[2])
    else:
        season = int(hist["season"].max())
        rnd = int(hist[hist["season"] == season]["round"].max()) + 1

    fixtures = fixtures_for(season, rnd)
    if not fixtures:
        # Normal between rounds: the upcoming round's rosters aren't published
        # yet. Nothing to project — exit cleanly so the daily train/commit
        # pipeline still succeeds. The 3-hourly predict run will emit
        # projection_inputs.json as soon as the fixtures/rosters appear.
        print(f"No fixtures yet for {season} R{rnd} — skipping projections.")
        return
    print(f"{season} R{rnd}: {len(fixtures)} fixtures")

    ph = build_placeholders(hist, fixtures, season, rnd)
    if ph.empty:
        print("No lineups resolved.", file=sys.stderr); sys.exit(1)

    # recompute leakage-safe features on history + placeholders, keep placeholders
    combo = pd.concat([hist, ph], ignore_index=True)
    combo = F.add_trailing(combo)
    combo = F.add_team_share(combo)
    combo = F.add_opponent_conceded(combo)
    combo = F.add_categoricals(combo)
    pred_rows = combo[(combo["season"] == season) & (combo["round"] == rnd)].copy()
    for c in feat:
        if c not in pred_rows:
            pred_rows[c] = np.nan

    # predict every target
    for t in TARGETS:
        pred_rows[f"exp_{t}"] = np.clip(models[t].predict(pred_rows[feat]), 0, None)

    # assemble JSON: per match -> per player expecteds, shares, dispersion
    out = {"season": season, "round": rnd,
           "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "dispersion": disp, "matches": []}
    for fx in fixtures:
        mp = pred_rows[pred_rows["match_id"] == fx["match_id"]]
        if mp.empty:
            continue
        teams = {}
        for tid in [fx["home_team_id"], fx["away_team_id"]]:
            tp = mp[mp["team_id"] == tid]
            tot = {t: float(tp[f"exp_{t}"].sum()) for t in TARGETS}
            players = []
            for _, r in tp.iterrows():
                exp = {t: round(float(r[f"exp_{t}"]), 3) for t in TARGETS}
                share = {t: round(exp[t] / tot[t], 5) if tot[t] > 0 else 0.0 for t in TARGETS}
                players.append({
                    "player_id": r["player_id"], "player": r["player"],
                    "position": r["position"], "role": r["role"],
                    "is_ruck": int(r["is_ruck"]),
                    "tog": round(float(r.get("timeOnGroundPercentage_r5", np.nan)) or 0, 1),
                    "exp": exp, "share": share,
                })
            team_points = round(tot["goals"] * 6 + tot["behinds"], 1)
            teams[tid] = {"team_id": tid, "team": tp["team"].iloc[0],
                          "team_total": {k: round(v, 2) for k, v in tot.items()},
                          "exp_points": team_points, "players": players}
        h, a = teams.get(fx["home_team_id"]), teams.get(fx["away_team_id"])
        if not (h and a):
            continue
        out["matches"].append({
            "match_id": fx["match_id"], "venue": fx["venue"], "date": fx["date"],
            "home": h, "away": a,
            "exp_total_points": round(h["exp_points"] + a["exp_points"], 1),
            "exp_supremacy": round(h["exp_points"] - a["exp_points"], 1),
        })

    os.makedirs("artifacts", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    np_players = sum(len(m["home"]["players"]) + len(m["away"]["players"]) for m in out["matches"])
    print(f"Wrote {OUT}: {len(out['matches'])} matches, {np_players} player projections")


if __name__ == "__main__":
    main()
