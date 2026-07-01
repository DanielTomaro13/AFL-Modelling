"""
Leakage-safe feature engineering for AFL player-stat models.

Every trailing feature uses only games STRICTLY BEFORE the current one
(shift(1) within player before any rolling), so a row never sees its own result.

Builds, per player-match:
  - trailing means (r3/r5/r10) + EWM of targets & core stats
  - trailing per-time-on-ground rates of targets (separates "played less" from "played worse")
  - season-to-date and career-to-date means
  - role/position encodings, is_ruck, kicking foot, home, venue, weather
  - experience (career games), days rest, interstate travel
  - player's trailing share of team total
  - opponent-conceded rollups, overall and split by role (what the opponent
    allows to rucks/mids/etc. — the matchup that actually applies to this player)

Output:
  data/processed/features.parquet
  artifacts/feature_cols.json   (model input column list)
"""
import os, json
import numpy as np
import pandas as pd

IN = "data/processed/player_match.parquet"
OUT = "data/processed/features.parquet"
COLS_OUT = "artifacts/feature_cols.json"

TARGETS = ["disposals", "goals", "kicks", "handballs", "marks", "tackles",
           "behinds", "totalClearances", "hitouts", "dreamTeamPoints"]

# stats we build trailing history on (targets + role-revealing context)
HIST_STATS = TARGETS + [
    "timeOnGroundPercentage", "contestedPossessions", "uncontestedPossessions",
    "centreBounceAttendances", "ruckContests", "inside50s", "marksInside50",
    "shotsAtGoal", "goalAssists", "scoreInvolvements", "metresGained",
    "freesFor", "freesAgainst", "centreClearances", "stoppageClearances",
    "contestedMarks", "intercepts", "pressureActs",
]
WINDOWS = (3, 5, 10)


def _roll(series_shifted, by, w):
    return series_shifted.groupby(by, sort=False).transform(
        lambda s: s.rolling(w, min_periods=1).mean())


def add_trailing(df):
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    pid = df["player_id"]
    g = df.groupby("player_id", sort=False)
    # accumulate all new columns then concat once (avoids frame fragmentation)
    cols = {}
    for stat in HIST_STATS:
        sh = g[stat].shift(1)
        for w in WINDOWS:
            cols[f"{stat}_r{w}"] = _roll(sh, pid, w)
        cols[f"{stat}_ewm"] = sh.groupby(pid, sort=False).transform(
            lambda s: s.ewm(halflife=3, min_periods=1).mean())
        cols[f"{stat}_career"] = sh.groupby(pid, sort=False).transform(
            lambda s: s.expanding(min_periods=1).mean())  # career-to-date, shifted
    # per-time-on-ground rates: a sub/injury-shortened game (~5% of rows) drags
    # raw trailing means down; the rate keeps output-per-minute-played clean
    tog = df["timeOnGroundPercentage"].clip(lower=25) / 100.0
    for stat in TARGETS:
        sh = (df[stat] / tog).groupby(pid, sort=False).shift(1)
        cols[f"{stat}_ptog_r5"] = _roll(sh, pid, 5)
        cols[f"{stat}_ptog_ewm"] = sh.groupby(pid, sort=False).transform(
            lambda s: s.ewm(halflife=3, min_periods=1).mean())
    # season-to-date mean of each target (expanding within player+season)
    for stat in TARGETS:
        sh = df.groupby(["player_id", "season"])[stat].shift(1)
        cols[f"{stat}_std"] = sh.groupby([df["player_id"], df["season"]],
                                         sort=False).transform(
            lambda s: s.expanding(min_periods=1).mean())
    cols["career_games"] = g.cumcount()
    cols["days_rest"] = g["date"].diff().dt.days.clip(0, 30)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def add_team_share(df):
    """Player's trailing share of their team's total for each target."""
    tt = df.groupby(["match_id", "team_id"])[TARGETS].transform("sum")
    cols = {}
    for stat in TARGETS:
        share = df[stat] / tt[stat].replace(0, np.nan)
        cols[f"{stat}_share_now"] = share
        sh = share.groupby(df["player_id"], sort=False).shift(1)
        cols[f"{stat}_share_r5"] = sh.groupby(df["player_id"], sort=False).transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def add_opponent_conceded(df):
    """How much the opponent typically concedes (trailing, leakage-safe)."""
    # team total per match + that team's opponent
    tt = (df.groupby(["match_id", "team_id"], as_index=False)[TARGETS].sum())
    # one row per (match, team) — guard against stray dup opponent/date labels
    link = (df[["match_id", "team_id", "opponent_id", "date"]]
            .drop_duplicates(subset=["match_id", "team_id"], keep="first"))
    tt = tt.merge(link, on=["match_id", "team_id"], how="left")
    # conceded by team T in a match = opponent team's total that match
    opp = tt.rename(columns={"team_id": "opp", **{t: f"conc_{t}" for t in TARGETS}})
    opp = opp[["match_id", "opp"] + [f"conc_{t}" for t in TARGETS]]
    conc = tt.merge(opp, left_on=["match_id", "opponent_id"],
                    right_on=["match_id", "opp"], how="left")
    conc = conc.sort_values(["team_id", "date"])
    for t in TARGETS:
        sh = conc.groupby("team_id")[f"conc_{t}"].shift(1)
        conc[f"opp_conc_{t}_r5"] = sh.groupby(conc["team_id"], sort=False).transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    keep = ["match_id", "team_id"] + [f"opp_conc_{t}_r5" for t in TARGETS]
    # join onto player rows by their OPPONENT (opponent's conceded rate)
    cm = (conc[keep].rename(columns={"team_id": "opponent_id"})
          .drop_duplicates(subset=["match_id", "opponent_id"], keep="first"))
    df = df.merge(cm, on=["match_id", "opponent_id"], how="left")
    return df.copy()  # de-fragment after the many column inserts


def _role_series(df):
    pos = df["position"].fillna("UNKNOWN")
    return pd.Series(np.select(
        [pos.str.contains("RUCK"),
         pos.str.contains("DEF"),
         pos.str.contains("MID"),
         pos.str.contains("FORWARD") | pos.str.contains("FWD")],
        ["RUCK", "DEF", "MID", "FWD"], default="OTHER"), index=df.index)


def add_opponent_conceded_role(df):
    """Opponent's trailing concessions to THIS player's role (matchup feature).
    Team-wide conceded hitouts say little to a ruck; conceded-to-rucks does."""
    role = _role_series(df)
    grp = (df.assign(_role=role)
           .groupby(["match_id", "opponent_id", "_role"], as_index=False)[TARGETS].sum()
           .rename(columns={"opponent_id": "conc_team",
                            **{t: f"rc_{t}" for t in TARGETS}}))
    md = df.drop_duplicates("match_id")[["match_id", "date"]]
    grp = grp.merge(md, on="match_id").sort_values(["conc_team", "_role", "date"])
    for t in TARGETS:
        sh = grp.groupby(["conc_team", "_role"])[f"rc_{t}"].shift(1)
        grp[f"opp_conc_role_{t}_r5"] = sh.groupby(
            [grp["conc_team"], grp["_role"]], sort=False).transform(
            lambda s: s.rolling(5, min_periods=1).mean())
    keep = ["match_id", "conc_team", "_role"] + [f"opp_conc_role_{t}_r5" for t in TARGETS]
    cm = grp[keep].drop_duplicates(subset=["match_id", "conc_team", "_role"], keep="first")
    df = (df.assign(_role=role)
          .merge(cm, left_on=["match_id", "opponent_id", "_role"],
                 right_on=["match_id", "conc_team", "_role"], how="left")
          .drop(columns=["conc_team", "_role"]))
    return df.copy()


def add_travel(df):
    """Interstate travel: playing in a state that isn't the team's home state."""
    home = df[pd.to_numeric(df["is_home"], errors="coerce") == 1]
    hs = home.groupby("team_id")["venue_state"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else None)
    home_state = df["team_id"].map(hs)
    df["travel"] = (df["venue_state"].notna() & home_state.notna()
                    & (df["venue_state"] != home_state)).astype("int8")
    return df


def add_categoricals(df):
    pos = df["position"].fillna("UNKNOWN")
    df["is_ruck"] = pos.str.contains("RUCK").astype("int8")
    df["role"] = _role_series(df)
    df["role_code"] = pd.Categorical(df["role"]).codes
    df["pos_code"] = pd.Categorical(pos).codes
    df["foot_right"] = (df["kicking_foot"] == "RIGHT").astype("int8")
    df["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype("int8")
    return df.copy()  # de-fragment


def main():
    df = pd.read_parquet(IN)
    df = add_trailing(df)
    df = add_team_share(df)
    df = add_opponent_conceded(df)
    df = add_opponent_conceded_role(df)
    df = add_categoricals(df)
    df = add_travel(df)
    df = df.sort_values(["date", "match_id", "team_id", "player_id"]).reset_index(drop=True)

    # assemble model feature column list (numeric, leakage-safe)
    feat = []
    for stat in HIST_STATS:
        feat += [f"{stat}_r{w}" for w in WINDOWS] + [f"{stat}_ewm", f"{stat}_career"]
    feat += [f"{t}_std" for t in TARGETS]
    feat += [f"{t}_share_r5" for t in TARGETS]
    feat += [f"opp_conc_{t}_r5" for t in TARGETS]
    feat += [f"opp_conc_role_{t}_r5" for t in TARGETS]
    feat += [f"{t}_ptog_r5" for t in TARGETS] + [f"{t}_ptog_ewm" for t in TARGETS]
    feat += ["career_games", "days_rest", "is_ruck", "role_code", "pos_code",
             "foot_right", "is_home", "age", "height_cm", "temp_c", "travel"]
    feat = [c for c in feat if c in df.columns]

    os.makedirs("artifacts", exist_ok=True)
    df.to_parquet(OUT, index=False)
    with open(COLS_OUT, "w") as f:
        json.dump(feat, f, indent=0)
    print(f"Wrote {OUT}: {len(df):,} rows, {len(feat)} model features")
    miss = df[feat].isna().mean().sort_values(ascending=False)
    print("Top missing-rate features:\n" + miss.head(6).round(3).to_string())


if __name__ == "__main__":
    main()
