"""Live bus lookup: find a stop by name and report the next buses and their delay.

"Delay" uses Seoul's own service standard. Trunk buses run on a scheduled
interval (``term``, minutes) rather than a fixed timetable, so a stop is
running late when the next bus is further away than one scheduled interval
(plus a small tolerance), and buses are *bunched* when two arrive much closer
together than scheduled.
"""
from __future__ import annotations

import json
import math
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

STOP_INDEX = Path(__file__).resolve().parents[1] / "data" / "reference" / "bus_stops.json"
LATE_TOLERANCE_MIN = 2.0       # waits up to interval + 2 min still count as on schedule
BUNCHING_RATIO = 0.35          # second bus closer than 35% of the interval -> bunched


def load_stop_index(path: Path = STOP_INDEX) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace(" ", "").replace(".", "").lower()


def search_stops(index: dict, query: str, limit: int = 30) -> list[dict]:
    """Physical stops (grouped by arsId) whose name contains the query.

    Each result: {arsId, stNm, routes: [{route_id, rtNm, term, ord, next_stop}]}.
    ``next_stop`` gives the direction of travel at that stop.
    """
    q = _norm(query)
    if not q:
        return []
    found: dict[str, dict] = {}
    for route_id, route in index["routes"].items():
        stops = route["stops"]
        for i, stop in enumerate(stops):
            if q not in _norm(stop["stNm"]) and q != stop["arsId"]:
                continue
            entry = found.setdefault(stop["arsId"], {"arsId": stop["arsId"],
                                                     "stNm": stop["stNm"], "routes": []})
            nxt = stops[i + 1]["stNm"] if i + 1 < len(stops) else "종점 (terminus)"
            entry["routes"].append({"route_id": route_id, "rtNm": route["rtNm"],
                                    "term": route["term"], "ord": stop["ord"],
                                    "next_stop": nxt})
    results = sorted(found.values(), key=lambda s: (not _norm(s["stNm"]).startswith(q),
                                                    s["stNm"], s["arsId"]))
    return results[:limit]


WALK_M_PER_MIN = 80           # average walking speed
NOMINATIM = "https://nominatim.openstreetmap.org/search"
SEOUL_VIEWBOX = "126.70,37.75,127.25,37.38"          # lon/lat box around Seoul + edges
USER_AGENT = "transit-delay-predictor/1.0 (portfolio demo app)"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_stops(index: dict, lat: float, lon: float, k: int = 5,
                  max_m: float = 2000) -> list[dict]:
    """The k closest physical stops (by arsId) on tracked routes, within max_m metres.

    Same shape as search_stops() results, plus distance_m, walk_min, lat, lon.
    """
    by_ars: dict[str, dict] = {}
    for route_id, route in index["routes"].items():
        stops = route["stops"]
        for i, stop in enumerate(stops):
            if stop.get("lat") is None:
                continue
            entry = by_ars.get(stop["arsId"])
            if entry is None:
                d = haversine_m(lat, lon, stop["lat"], stop["lon"])
                if d > max_m:
                    continue
                entry = by_ars[stop["arsId"]] = {
                    "arsId": stop["arsId"], "stNm": stop["stNm"], "lat": stop["lat"],
                    "lon": stop["lon"], "distance_m": round(d),
                    "walk_min": max(1, round(d / WALK_M_PER_MIN)), "routes": []}
            nxt = stops[i + 1]["stNm"] if i + 1 < len(stops) else "종점 (terminus)"
            entry["routes"].append({"route_id": route_id, "rtNm": route["rtNm"],
                                    "term": route["term"], "ord": stop["ord"],
                                    "next_stop": nxt})
    return sorted(by_ars.values(), key=lambda s: s["distance_m"])[:k]


def geocode(query: str, timeout: float = 15,
            lang: str = "ko") -> tuple[float, float, str] | None:
    """Place name or address -> (lat, lon, label) via OpenStreetMap Nominatim.

    Bounded to the Seoul area. Callers must cache results: Nominatim's usage
    policy allows at most one request per second.
    """
    params = urllib.parse.urlencode({
        "q": query, "format": "jsonv2", "limit": 1, "countrycodes": "kr",
        "viewbox": SEOUL_VIEWBOX, "bounded": 1, "accept-language": lang})
    req = urllib.request.Request(f"{NOMINATIM}?{params}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            results = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not results:
        return None
    top = results[0]
    label = ", ".join(top.get("display_name", query).split(", ")[:3])
    return float(top["lat"]), float(top["lon"]), label


@dataclass
class StopStatus:
    route: str
    term_min: int
    next_min: float | None
    second_min: float | None
    status: str                 # "On schedule" | "Delayed" | "Bunched" | "No bus" | message
    delay_min: float
    crowding: str
    message: str


CROWDING = {"3": "Seats available", "4": "Standing room", "5": "Crowded", "0": "Unknown"}


def _minutes(seconds) -> float | None:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return None
    return round(s / 60, 1) if s > 0 else None


def summarize_stop(row: dict, term: int) -> StopStatus:
    """Turn one getArrInfoByRouteAll row (one stop on one route) into a status."""
    nxt, second = _minutes(row.get("traTime1")), _minutes(row.get("traTime2"))
    msg = (row.get("arrmsg1") or "").strip()
    crowd = CROWDING.get(str(row.get("reride_Num1") or "0"), "Unknown")
    if row.get("isArrive1") == "1" or msg.startswith("곧 도착"):
        nxt = 0.0
    if nxt is None:
        return StopStatus(row.get("rtNm", "?"), term, None, second,
                          "No bus" if not msg else msg, 0.0, crowd, msg)
    late = bool(term) and nxt > term + LATE_TOLERANCE_MIN
    delay = round(nxt - term, 1) if late else 0.0
    if late:
        status = "Delayed"
    elif second is not None and term and (second - nxt) < term * BUNCHING_RATIO:
        status = "Bunched"
    else:
        status = "On schedule"
    return StopStatus(row.get("rtNm", "?"), term, nxt, second, status, delay, crowd, msg)
