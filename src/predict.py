"""
Produce per-player projection inputs for one round -> artifacts/projection_inputs.json

This is the bridge the TS Monte-Carlo engine (in AFL-23-0) consumes. For each
fixture it emits, per player: expected value of every target, the player's share
of the team total, and the dispersion% — all derived empirically from the models.

Lineups: confirmed roster if available, else a proxy = each team's most recent XVIII.

Usage:
  python src/predict.py <season> <round>        # e.g. 2026 15
  python src/predict.py                          # auto: next round that isn't fully played
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


def _roster_players(side):
    """Named players from a matchRoster side (confirmed/provisional team), or [].
    Emergencies are excluded — they inflate team totals and dilute shares."""
    out = []
    for pos in (side or {}).get("positions") or []:
        pl = (pos or {}).get("player") or {}
        nm = pl.get("playerName") or {}
        if (pos or {}).get("position") == "EMERG":
            continue
        if pl.get("playerId"):
            out.append({"player_id": pl["playerId"],
                        "named_pos": pos.get("position"),
                        "given_name": nm.get("givenName"),
                        "surname": nm.get("surname"),
                        "jumper": pl.get("playerJumperNumber")})
    return out


def fixtures_for(season, rnd, force=False):
    rosters = api.match_rosters_round(season, rnd, force=force)
    out = []
    for item in (rosters or []):
        mt = (item or {}).get("match") or {}
        vn = (item or {}).get("venue") or {}
        mr = (item or {}).get("matchRoster") or {}
        wx = (mr.get("weather") or {}) if isinstance(mr, dict) else {}
        if not mt.get("homeTeamId"):
            continue
        out.append({
            "match_id": mt.get("matchId"),
            "home_team_id": mt.get("homeTeamId"),
            "away_team_id": mt.get("awayTeamId"),
            "venue": vn.get("name"), "venue_state": vn.get("state"),
            "date": mt.get("utcStartTime") or mt.get("date"),
            "status": mt.get("status"),
            "temp_c": wx.get("tempInCelsius"), "weather": wx.get("weatherType"),
            "home_roster": _roster_players(mr.get("homeTeam")),
            "away_roster": _roster_players(mr.get("awayTeam")),
            "home_status": (mr.get("homeTeam") or {}).get("teamStatus"),
            "away_status": (mr.get("awayTeam") or {}).get("teamStatus"),
        })
    if out:
        return out
    # Roster feed not published yet (teams are named ~Thu). Fall back to the
    # schedule feed so the round can still be projected with proxy lineups —
    # empty rosters below make build_placeholders() use each team's last XVIII.
    return fixtures_from_items(season, rnd, force=force)


def fixtures_from_items(season, rnd, force=False):
    """Fixture stubs from the schedule feed (no named teams / weather)."""
    items = api.match_items_round(season, rnd, force=force)
    out = []
    for it in ((items or {}).get("items") or []):
        mt = (it or {}).get("match") or {}
        vn = (it or {}).get("venue") or {}
        if not mt.get("homeTeamId"):
            continue
        out.append({
            "match_id": mt.get("matchId"),
            "home_team_id": mt.get("homeTeamId"),
            "away_team_id": mt.get("awayTeamId"),
            "venue": vn.get("name"), "venue_state": vn.get("state"),
            "date": mt.get("utcStartTime") or mt.get("date"),
            "status": mt.get("status"),
            "temp_c": None, "weather": None,
            "home_roster": [], "away_roster": [],
            "home_status": None, "away_status": None,
        })
    return out


def upcoming_round(season, base, lookahead=8):
    """The round to project, chosen from the SCHEDULE rather than from results
    ingestion: the first round at or after `base` that still has a match which
    hasn't CONCLUDED. This advances to the next round the moment the current one
    finishes, instead of keying off `max(played round in history) + 1` — which
    leaves the site stuck re-projecting an already-played round for days until
    that round's results are ingested into history.

    Fetched with force=True so round status is live (not a stale on-disk cache).
    Returns (round_number, fixtures) or (None, []) if nothing upcoming is found.
    """
    for rnd in range(base, base + lookahead):
        fx = fixtures_for(season, rnd, force=True)
        if not fx:
            continue
        if any((f.get("status") or "").upper() != "CONCLUDED" for f in fx):
            return rnd, fx
    return None, []


def proxy_lineup(hist, team_id, before_date):
    """Players from this team's most recent match strictly before `before_date`."""
    h = hist[(hist["team_id"] == team_id) & (hist["date"] < before_date)]
    if h.empty:
        return h
    last = h["date"].max()
    return h[h["date"] == last]


def build_placeholders(hist, fixtures, season, rnd):
    """One placeholder row per projected player. Uses the CONFIRMED named team
    when the roster payload has one (>=18 players), else the most-recent-XVIII
    proxy. Returns (frame, meta) where meta records lineup source + named
    positions for the output JSON."""
    latest = (hist.sort_values("date").groupby("player_id").tail(1)
              .set_index("player_id"))
    meta = {"lineup_status": {}, "named_pos": {}}
    rows = []
    for fx in fixtures:
        bd = pd.to_datetime(fx["date"], utc=True) if fx["date"] else hist["date"].max()
        for side, tid, oid in [("home", fx["home_team_id"], fx["away_team_id"]),
                               ("away", fx["away_team_id"], fx["home_team_id"])]:
            base = {
                "season": season, "round": rnd, "round_id": api.round_provider(season, rnd),
                "match_id": fx["match_id"], "date": bd,
                "venue": fx["venue"], "venue_state": fx["venue_state"],
                "temp_c": fx.get("temp_c", np.nan), "weather": fx.get("weather"),
                "is_home": 1 if side == "home" else 0,
                "team_id": tid, "opponent_id": oid,
                "opponent": None, "opponent_abbr": None,
                "outcome": None, "team_score": np.nan, "opp_score": np.nan, "margin": np.nan,
            }
            roster = fx.get(f"{side}_roster") or []
            if len(roster) >= 18:
                status = (fx.get(f"{side}_status") or "").upper()
                meta["lineup_status"][tid] = (
                    "confirmed" if "CONFIRMED" in status else "provisional")
                team_name = team_abbr = None
                if not hist[hist["team_id"] == tid].empty:
                    tl = hist[hist["team_id"] == tid].iloc[-1]
                    team_name, team_abbr = tl["team"], tl["team_abbr"]
                for rp in roster:
                    pid = rp["player_id"]
                    meta["named_pos"][pid] = rp.get("named_pos")
                    h = latest.loc[pid] if pid in latest.index else None
                    rows.append({
                        **base, "player_id": pid,
                        "player": (f"{rp.get('given_name','') or ''} "
                                   f"{rp.get('surname','') or ''}").strip(),
                        "given_name": rp.get("given_name"), "surname": rp.get("surname"),
                        # keep the HISTORICAL position so categorical codes stay
                        # aligned with training; roster field position goes to
                        # the output JSON as named_pos instead
                        "position": h["position"] if h is not None else None,
                        "kicking_foot": h["kicking_foot"] if h is not None else None,
                        "age": h["age"] if h is not None else np.nan,
                        "height_cm": h["height_cm"] if h is not None else np.nan,
                        "jumper": rp.get("jumper"),
                        "team": team_name, "team_abbr": team_abbr,
                    })
            else:
                meta["lineup_status"][tid] = "proxy"
                lu = proxy_lineup(hist, tid, bd)
                for _, p in lu.iterrows():
                    rows.append({
                        **base, "player_id": p["player_id"], "player": p["player"],
                        "given_name": p["given_name"], "surname": p["surname"],
                        "position": p["position"], "kicking_foot": p["kicking_foot"],
                        "age": p["age"], "height_cm": p["height_cm"], "jumper": p["jumper"],
                        "team": p["team"], "team_abbr": p["team_abbr"],
                    })
    # align to history columns; stat columns not set here become NaN placeholders
    return pd.DataFrame(rows).reindex(columns=hist.columns), meta


def main():
    import joblib
    bundle = joblib.load(MODELS)
    models, feat = bundle["models"], bundle["features"]
    disp = json.load(open(DISP))
    hist = pd.read_parquet(HIST)

    if len(sys.argv) >= 3:
        season, rnd = int(sys.argv[1]), int(sys.argv[2])
        fixtures = fixtures_for(season, rnd)
    else:
        season = int(hist["season"].max())
        # Start from the last round present in history (which may still be in
        # progress) and walk forward to the first round that isn't fully played.
        base = int(hist[hist["season"] == season]["round"].max())
        rnd, fixtures = upcoming_round(season, base)
        if rnd is None:
            # End of season / no scheduled fixtures ahead. Leave the last
            # projection_inputs.json in place and exit cleanly.
            print("No upcoming round with fixtures — skipping projections.")
            return

    if not fixtures:
        print(f"No fixtures yet for {season} R{rnd} — skipping projections.")
        return
    print(f"{season} R{rnd}: {len(fixtures)} fixtures")

    ph, meta = build_placeholders(hist, fixtures, season, rnd)
    if ph.empty:
        print("No lineups resolved.", file=sys.stderr); sys.exit(1)
    from collections import Counter
    print("lineups:", dict(Counter(meta["lineup_status"].values())))

    # recompute leakage-safe features on history + placeholders, keep placeholders
    combo = pd.concat([hist, ph], ignore_index=True)
    combo = F.add_trailing(combo)
    combo = F.add_team_share(combo)
    combo = F.add_opponent_conceded(combo)
    combo = F.add_opponent_conceded_role(combo)
    combo = F.add_categoricals(combo)
    combo = F.add_travel(combo)
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
                # per-player predictive spread from the calibrated sigma model
                sigma = {}
                for t in TARGETS:
                    d = disp[t]
                    if d["type"] == "poisson":
                        s = float(np.sqrt(max(d.get("phi", 1.0), 0.5) * max(exp[t], 1e-3)))
                    else:
                        s = d.get("sigma_a", 0.0) + d.get("sigma_b", 0.0) * float(np.sqrt(exp[t]))
                        s = max(d.get("sigma_floor", 0.2), s) if "sigma_a" in d else d["resid_std"]
                    sigma[t] = round(s, 3)
                tv = r.get("timeOnGroundPercentage_r5")
                players.append({
                    "player_id": r["player_id"], "player": r["player"],
                    "position": r["position"], "role": r["role"],
                    "named_pos": meta["named_pos"].get(r["player_id"]),
                    "is_ruck": int(r["is_ruck"]),
                    "tog": 0.0 if pd.isna(tv) else round(float(tv), 1),
                    "exp": exp, "share": share, "sigma": sigma,
                })
            team_points = round(tot["goals"] * 6 + tot["behinds"], 1)
            tnames = tp["team"].dropna()
            teams[tid] = {"team_id": tid,
                          "team": tnames.iloc[0] if not tnames.empty else tid,
                          "lineup": meta["lineup_status"].get(tid, "proxy"),
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
