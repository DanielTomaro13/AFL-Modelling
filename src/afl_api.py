"""
Thin client for the public AFL StatsPro / CFS feeds (api.afl.com.au).

Mirrors what the AFL site itself does:
  POST /cfs/afl/WMCTok          -> short-lived anonymous x-media-mis-token
  GET  /statspro/... , /cfs/... -> data, with the token in a header

Only factual sporting data (player stat lines, fixtures) is read. Every fetch
is cached to data/raw/ so re-runs are free and the historical sweep is resumable.
No API key is required.
"""
import json, os, time, hashlib, http.client, urllib.request, urllib.error
from urllib.parse import urlencode

BASE = "https://api.afl.com.au"
UA = os.environ.get(
    "AFL_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
RAW_DIR = "data/raw"
MIN_INTERVAL = 0.35  # polite throttle between live requests (s)

_token = None
_last_req = 0.0


def _mint_token():
    # undici/akamai needs a non-empty body for this POST; a single space works.
    req = urllib.request.Request(
        f"{BASE}/cfs/afl/WMCTok", data=b" ", method="POST",
        headers={"User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["token"]


def _throttle():
    global _last_req
    wait = _last_req + MIN_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_req = time.time()


def _live_get(path, params):
    global _token
    url = f"{BASE}{path}"
    if params:
        url += "?" + urlencode(params)
    for attempt in range(5):
        if _token is None:
            _token = _mint_token()
        _throttle()
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "x-media-mis-token": _token,
            "Origin": "https://www.afl.com.au",
            "Referer": "https://www.afl.com.au/",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                _token = None       # refresh token and retry
            elif e.code == 404:
                return None         # round/match doesn't exist
            time.sleep(1.2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException, json.JSONDecodeError):
            # http.client.HTTPException covers IncompleteRead — a chunked
            # response truncated mid-stream by the server. Transient: retry.
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"AFL API failed after retries: {url}")


def get(path, params=None, cache_key=None, force=False):
    """Cached GET. cache_key (relative to data/raw) controls the on-disk path."""
    if cache_key:
        fp = os.path.join(RAW_DIR, cache_key)
        if not force and os.path.exists(fp):
            with open(fp) as f:
                return json.load(f)
        data = _live_get(path, params)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as f:
            json.dump(data, f)
        return data
    return _live_get(path, params)


# ── provider-id helpers (Toyota AFL Premiership competition code is "014") ──
def season_provider(year):           return f"CD_S{year}014"
def round_provider(year, rnd):       return f"CD_R{year}014{rnd:02d}"


def players_stats_round(year, rnd, force=False):
    rid = round_provider(year, rnd)
    return get(f"/statspro/playersStats/rounds/{rid}",
               cache_key=f"{year}/players_r{rnd:02d}.json", force=force)


def match_rosters_round(year, rnd, force=False):
    rid = round_provider(year, rnd)
    return get(f"/cfs/afl/matchRosters/round/{rid}", params={"minimal": "true"},
               cache_key=f"{year}/rosters_r{rnd:02d}.json", force=force)


def match_items_round(year, rnd, force=False):
    """Round fixtures from the schedule feed — match id, teams, venue, date,
    status. Available as soon as the fixture is published (well before named
    teams appear in matchRosters), so it lets us project a round with proxy
    lineups before rosters exist."""
    rid = round_provider(year, rnd)
    return get(f"/cfs/afl/matchItems/round/{rid}",
               cache_key=f"{year}/items_r{rnd:02d}.json", force=force)
