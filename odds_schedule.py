import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

ODDS_API_URL_TEMPLATE = (
    "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
)

# Light mapping between The Odds API team names and nflfastR-style abbreviations.
ODDS_TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

# Older or alternate abbreviations that should map to the modern nflfastR style.
TEAM_ABBR_FIXES = {
    "ARI": "ARI",
    "BLT": "BAL",
    "BAL": "BAL",
    "CLV": "CLE",
    "CLE": "CLE",
    "GBP": "GB",
    "GNB": "GB",
    "GB": "GB",
    "HST": "HOU",
    "HOU": "HOU",
    "JAC": "JAX",
    "JAX": "JAX",
    "KAN": "KC",
    "KC": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "SD": "LAC",
    "LAC": "LAC",
    "OAK": "LV",
    "LVR": "LV",
    "LV": "LV",
    "NOR": "NO",
    "NO": "NO",
    "NWE": "NE",
    "NE": "NE",
    "SFO": "SF",
    "SF": "SF",
    "STL": "LAR",
    "TAM": "TB",
    "TB": "TB",
    "WFT": "WAS",
    "WSH": "WAS",
    "WAS": "WAS",
}

# Common column names found in historical line files that should be normalized.
HISTORICAL_NAME_MAP = {
    "ml_home": "home_moneyline",
    "ml_away": "away_moneyline",
    "moneyline_home": "home_moneyline",
    "moneyline_away": "away_moneyline",
    "home_ml": "home_moneyline",
    "away_ml": "away_moneyline",
    "spread": "spread_line",
    "spread_home": "spread_line",
    "total": "total_line",
    "game_total": "total_line",
    "over": "over_odds",
    "under": "under_odds",
}

ODDS_VALUE_COLUMNS = [
    "home_moneyline",
    "away_moneyline",
    "spread_line",
    "home_spread_odds",
    "away_spread_odds",
    "total_line",
    "over_odds",
    "under_odds",
]


def _normalize_team(name: str) -> Optional[str]:
    if name is None:
        return None
    if name in ODDS_TEAM_NAME_TO_ABBR:
        return ODDS_TEAM_NAME_TO_ABBR[name]
    if isinstance(name, str):
        upper = name.upper()
        if upper in TEAM_ABBR_FIXES:
            return TEAM_ABBR_FIXES[upper]
    return name


def _select_bookmaker(
    bookmakers: Sequence[Dict], preference: Optional[str]
) -> Optional[Dict]:
    if not bookmakers:
        return None
    if preference:
        for book in bookmakers:
            if book.get("key", "").lower() == preference.lower():
                return book
    return bookmakers[0]


def _find_market(bookmaker: Dict, market_key: str) -> Optional[Dict]:
    for market in bookmaker.get("markets", []):
        if market.get("key") == market_key:
            return market
    return None


def _find_outcome(market: Dict, target: str) -> Optional[Dict]:
    for outcome in market.get("outcomes", []):
        if outcome.get("name") == target:
            return outcome
    return None


def _odds_json_to_frame(
    events: Iterable[Dict], bookmaker_preference: Optional[str]
) -> pd.DataFrame:
    rows: List[Dict] = []
    for event in events:
        home_full = event.get("home_team")
        teams = event.get("teams", [])
        away_full = next((t for t in teams if t != home_full), None)
        if not home_full or not away_full:
            continue

        bookmaker = _select_bookmaker(event.get("bookmakers", []), bookmaker_preference)
        if not bookmaker:
            continue

        row: Dict = {
            "home_team": _normalize_team(home_full),
            "away_team": _normalize_team(away_full),
            "commence_time": event.get("commence_time"),
            "odds_source": "odds_api",
        }

        # Moneylines
        h2h = _find_market(bookmaker, "h2h")
        if h2h:
            home_ml = _find_outcome(h2h, home_full)
            away_ml = _find_outcome(h2h, away_full)
            row["home_moneyline"] = home_ml.get("price") if home_ml else None
            row["away_moneyline"] = away_ml.get("price") if away_ml else None

        # Spreads
        spreads = _find_market(bookmaker, "spreads")
        if spreads:
            home_spread = _find_outcome(spreads, home_full)
            away_spread = _find_outcome(spreads, away_full)
            if home_spread:
                row["spread_line"] = home_spread.get("point")
                row["home_spread_odds"] = home_spread.get("price")
            if away_spread:
                row.setdefault("spread_line", away_spread.get("point"))
                row["away_spread_odds"] = away_spread.get("price")

        # Totals
        totals = _find_market(bookmaker, "totals")
        if totals:
            over = _find_outcome(totals, "Over")
            under = _find_outcome(totals, "Under")
            if over:
                row["total_line"] = over.get("point")
                row["over_odds"] = over.get("price")
            if under:
                row.setdefault("total_line", under.get("point"))
                row["under_odds"] = under.get("price")

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["home_team", "away_team"])
    return pd.DataFrame(rows)


def _fetch_latest_odds(
    api_key: str,
    sport: str,
    regions: str,
    markets: str,
    odds_format: str,
    bookmaker_preference: Optional[str],
) -> pd.DataFrame:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    url = ODDS_API_URL_TEMPLATE.format(sport=sport)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return _odds_json_to_frame(data, bookmaker_preference)


def _prepare_historical_lines(
    historical_csv: str, key_cols: Sequence[str]
) -> Optional[pd.DataFrame]:
    if not historical_csv or not os.path.exists(historical_csv):
        return None
    hist = pd.read_csv(historical_csv)
    rename_map = {k: v for k, v in HISTORICAL_NAME_MAP.items() if k in hist.columns}
    hist = hist.rename(columns=rename_map)
    missing_keys = [col for col in key_cols if col not in hist.columns]
    if missing_keys:
        return None
    hist["home_team"] = hist["home_team"].map(_normalize_team)
    hist["away_team"] = hist["away_team"].map(_normalize_team)
    keep_cols = list(
        dict.fromkeys(list(key_cols) + [c for c in ODDS_VALUE_COLUMNS if c in hist.columns])
    )
    return hist[keep_cols]


def fetch_schedule_odds(
    schedule: pd.DataFrame,
    api_key: str,
    *,
    historical_csv: Optional[str] = None,
    sport: str = "americanfootball_nfl",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    bookmaker_preference: Optional[str] = "pinnacle",
) -> pd.DataFrame:
    """
    Enrich a schedule DataFrame with the latest odds from The Odds API and
    backfill any missing games from a historical CSV.

    Parameters
    ----------
    schedule:
        DataFrame that must contain at least season, week, home_team, and away_team.
    api_key:
        The Odds API key.  If empty, only historical lines will be used.
    historical_csv:
        Optional CSV with columns season, week, home_team, away_team, and any of the
        ODDS_VALUE_COLUMNS.  Used to backfill missing API rows.
    """

    required = {"season", "week", "home_team", "away_team"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule is missing required columns: {sorted(missing)}")

    schedule = schedule.copy()
    schedule["home_team"] = schedule["home_team"].map(_normalize_team)
    schedule["away_team"] = schedule["away_team"].map(_normalize_team)
    key_cols: List[str] = ["season", "week", "home_team", "away_team"]

    latest = pd.DataFrame(columns=["home_team", "away_team"])
    if api_key:
        try:
            latest = _fetch_latest_odds(
                api_key,
                sport=sport,
                regions=regions,
                markets=markets,
                odds_format=odds_format,
                bookmaker_preference=bookmaker_preference,
            )
        except Exception as exc:
            warnings.warn(f"Failed to fetch odds from API: {exc}")
            latest = pd.DataFrame(columns=key_cols)
    else:
        warnings.warn("No Odds API key provided; relying on historical lines only.")

    latest = latest.dropna(subset=["home_team", "away_team"])
    if not latest.empty and "commence_time" in latest.columns:
        latest = latest.sort_values("commence_time").drop_duplicates(
            subset=["home_team", "away_team"], keep="last"
        )
    latest_cols = ["home_team", "away_team"] + [
        c for c in ODDS_VALUE_COLUMNS + ["odds_source", "commence_time"] if c in latest.columns
    ]
    merged = schedule.merge(
        latest[latest_cols],
        on=["home_team", "away_team"],
        how="left",
    )

    for col in ODDS_VALUE_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    hist = _prepare_historical_lines(historical_csv, key_cols) if historical_csv else None
    if hist is not None and not hist.empty:
        merged = merged.merge(
            hist,
            on=key_cols,
            how="left",
            suffixes=("", "_hist"),
        )
        filled_from_hist = pd.Series(False, index=merged.index)
        for col in ODDS_VALUE_COLUMNS:
            hist_col = f"{col}_hist"
            if hist_col in merged.columns:
                mask = merged[col].isna() & merged[hist_col].notna()
                if mask.any():
                    merged.loc[mask, col] = merged.loc[mask, hist_col]
                    filled_from_hist = filled_from_hist | mask
                merged = merged.drop(columns=[hist_col])
        if "odds_source" not in merged.columns:
            merged["odds_source"] = pd.Series(pd.NA, index=merged.index)
        merged.loc[filled_from_hist & merged["odds_source"].isna(), "odds_source"] = "historical"
    else:
        if "odds_source" not in merged.columns:
            merged["odds_source"] = pd.Series(pd.NA, index=merged.index)

    merged["odds_source"] = merged["odds_source"].fillna("odds_api")
    still_missing = merged[ODDS_VALUE_COLUMNS].isna().all(axis=1)
    merged.loc[still_missing, "odds_source"] = "missing"
    return merged


__all__ = ["fetch_schedule_odds"]
