"""
Ingest AFL men's Premiership player stat lines from the AFL StatsPro feed.

For each (season, round):
  matchRosters/round/{id}  -> fixture context: home/away, venue, weather, date
  playersStats/rounds/{id} -> one row per player with the full `totals` block

Output:
  data/processed/player_match.parquet  (one row per player per match)

Env:
  AFL_SEASONS  comma list of years (default 2015..CURRENT_YEAR)
  AFL_MAX_ROUND  highest round number to probe per season (default 30; covers finals)
"""
import os, re, sys, datetime
import pandas as pd
import afl_api as api

OUT = "data/processed/player_match.parquet"
CURRENT_YEAR = 2026
MAX_ROUND = int(os.environ.get("AFL_MAX_ROUND", "30"))

# the 10 modelled targets (kept first for clarity)
TARGETS = ["disposals", "goals", "kicks", "handballs", "marks", "tackles",
           "behinds", "totalClearances", "hitouts", "dreamTeamPoints"]

# extra `totals` fields kept as features / decomposition inputs
EXTRA_TOTALS = [
    "timeOnGroundPercentage", "centreClearances", "stoppageClearances",
    "contestedMarks", "marksInside50", "marksOnLead", "interceptMarks",
    "effectiveDisposals", "disposalEfficiency", "effectiveKicks", "kickEfficiency",
    "kickToHandballRatio", "goalAssists", "goalAccuracy", "shotsAtGoal",
    "hitoutsToAdvantage", "hitoutWinPercentage", "hitoutToAdvantageRate", "ruckContests",
    "centreBounceAttendances", "groundBallGets", "f50GroundBallGets",
    "contestedPossessions", "uncontestedPossessions", "totalPossessions",
    "contestedPossessionRate", "inside50s", "rebound50s", "metresGained",
    "intercepts", "spoils", "onePercenters", "pressureActs", "tacklesInside50",
    "freesFor", "freesAgainst", "clangers", "turnovers", "scoreInvolvements",
    "scoreLaunches", "ratingPoints", "bounces", "kickins", "kickinsPlayon",
]

RESULT_RE = re.compile(r"^\s*([WLD])\s+(\d+)\s*-\s*(\d+)")


def parse_result(s):
    """'W 136-49' -> (outcome, team_score, opp_score, margin)."""
    if not isinstance(s, str):
        return (None, None, None, None)
    m = RESULT_RE.match(s)
    if not m:
        return (None, None, None, None)
    out, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    return (out, a, b, a - b)


def fixture_map(rosters):
    """team_id -> fixture context dict, from a matchRosters_round payload."""
    out = {}
    if not isinstance(rosters, list):
        return out
    for item in rosters:
        mt = (item or {}).get("match") or {}
        vn = (item or {}).get("venue") or {}
        mr = (item or {}).get("matchRoster") or {}
        wx = (mr.get("weather") or {}) if isinstance(mr, dict) else {}
        home, away = mt.get("homeTeamId"), mt.get("awayTeamId")
        ctx_common = {
            "match_id": mt.get("matchId"),
            "date": mt.get("utcStartTime") or mt.get("date"),
            "venue": vn.get("name"), "venue_state": vn.get("state"),
            "temp_c": wx.get("tempInCelsius"), "weather": wx.get("weatherType"),
        }
        if home:
            out[home] = {**ctx_common, "is_home": 1, "opponent_id_fix": away}
        if away:
            out[away] = {**ctx_common, "is_home": 0, "opponent_id_fix": home}
    return out


def parse_round(year, rnd):
    players = api.players_stats_round(year, rnd)
    if not players or not players.get("players"):
        return []
    fix = fixture_map(api.match_rosters_round(year, rnd))
    round_id = api.round_provider(year, rnd)
    rows = []
    for p in players["players"]:
        if (p.get("gamesPlayed") or 0) < 1:
            continue
        d = p.get("playerDetails") or {}
        tm = p.get("team") or {}
        op = p.get("opponent") or {}
        tot = p.get("totals") or {}
        outcome, ts, os_, margin = parse_result(p.get("result"))
        ctx = fix.get(tm.get("teamId"), {})
        row = {
            "season": year, "round": rnd, "round_id": round_id,
            "match_id": ctx.get("match_id"), "date": ctx.get("date"),
            "venue": ctx.get("venue"), "venue_state": ctx.get("venue_state"),
            "temp_c": ctx.get("temp_c"), "weather": ctx.get("weather"),
            "is_home": ctx.get("is_home"),
            "player_id": p.get("playerId"),
            "player": f"{d.get('givenName','')} {d.get('surname','')}".strip(),
            "given_name": d.get("givenName"), "surname": d.get("surname"),
            "position": d.get("position"), "kicking_foot": d.get("kickingFoot"),
            "age": d.get("age"), "height_cm": d.get("heightCm"),
            "jumper": d.get("jumperNumber"),
            "team_id": tm.get("teamId"), "team": tm.get("teamName"),
            "team_abbr": tm.get("teamAbbr"),
            "opponent_id": op.get("teamId"), "opponent": op.get("teamName"),
            "opponent_abbr": op.get("teamAbbr"),
            "outcome": outcome, "team_score": ts, "opp_score": os_, "margin": margin,
        }
        for f in TARGETS + EXTRA_TOTALS:
            row[f] = tot.get(f)
        rows.append(row)
    return rows


def main():
    seasons = os.environ.get("AFL_SEASONS")
    if seasons:
        years = [int(y) for y in seasons.split(",") if y.strip()]
    else:
        years = list(range(2015, CURRENT_YEAR + 1))

    all_rows = []
    for year in years:
        season_rows = 0
        empty_streak = 0
        for rnd in range(1, MAX_ROUND + 1):
            rows = parse_round(year, rnd)
            if rows:
                all_rows.extend(rows)
                season_rows += len(rows)
                empty_streak = 0
            else:
                empty_streak += 1
                # 3 empty rounds in a row past round 5 => season is over
                if empty_streak >= 3 and rnd > 5:
                    break
        print(f"  {year}: {season_rows} player-match rows")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data ingested.", file=sys.stderr)
        sys.exit(1)

    # numeric coercion for stat columns
    for c in TARGETS + EXTRA_TOTALS + ["team_score", "opp_score", "margin",
                                       "age", "height_cm", "temp_c"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.sort_values(["date", "match_id", "team_id", "player_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}: {len(df):,} rows, {df['season'].nunique()} seasons, "
          f"{df['player_id'].nunique():,} players, "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    # quick target sanity
    print(df[TARGETS].describe().loc[["mean", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
