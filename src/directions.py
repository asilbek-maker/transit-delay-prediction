"""Door-to-door directions: public transit (ODsay) plus car / walk / bike (OSRM).

Transit routes come from ODsay's searchPubTransPathT in English (lang=1). Each bus
leg carries Seoul's own route ID (busLocalBlID) and stop ID (startLocalStationID),
which lets the app attach live arrival/delay status from Seoul's bus API.

Car, walking and cycling times come from the FOSSGIS OSRM servers (OpenStreetMap
data, no key). Car times assume free-flowing traffic.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

ODSAY_URL = "https://api.odsay.com/v1/api/searchPubTransPathT"
# The ODsay key is registered for the web platform with this URI; requests must
# identify themselves with it (Referer) or authentication fails.
APP_URL = "https://transit-delay-prediction-app.streamlit.app"
OSRM_URL = "https://routing.openstreetmap.de/{profile}/route/v1/driving/{a};{b}?overview=false"
OSRM_PROFILES = {"car": "routed-car", "walk": "routed-foot", "bike": "routed-bike"}
USER_AGENT = "transit-delay-predictor/1.0 (portfolio demo app)"

TRAFFIC = {1: "subway", 2: "bus", 3: "walk"}


class DirectionsError(RuntimeError):
    """ODsay returned an error (e.g. start and end closer than 700 m)."""


@dataclass
class Leg:
    kind: str                      # "walk" | "bus" | "subway"
    minutes: int
    distance_m: int = 0
    name: str = ""                 # bus number or subway line name
    start: str = ""
    end: str = ""
    start_kor: str = ""
    stops: int = 0
    interval_min: int | None = None          # scheduled headway (bus)
    seoul_route_id: str | None = None        # busLocalBlID -> Seoul bus API
    seoul_stop_id: str | None = None         # startLocalStationID -> Seoul bus API
    start_xy: tuple[float, float] | None = None   # (lat, lon)
    end_xy: tuple[float, float] | None = None


@dataclass
class TransitRoute:
    total_min: int
    fare_krw: int
    bus_transfers: int
    subway_transfers: int
    walk_min: int
    legs: list[Leg] = field(default_factory=list)

    @property
    def transfers(self) -> int:
        return max(0, self.bus_transfers + self.subway_transfers - 1)

    @property
    def modes(self) -> str:
        kinds = {leg.kind for leg in self.legs if leg.kind != "walk"}
        return " + ".join(k.title() for k in ("bus", "subway") if k in kinds) or "Walk"


def _xy(lat, lon) -> tuple[float, float] | None:
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def parse_odsay(payload: dict, limit: int = 5) -> list[TransitRoute]:
    """ODsay JSON -> routes (raises DirectionsError on API errors)."""
    err = payload.get("error")
    if err:
        if isinstance(err, list):
            err = err[0] if err else {}
        raise DirectionsError(f"{err.get('code', '')}: {err.get('msg') or err.get('message', '')}")
    routes = []
    for path in (payload.get("result") or {}).get("path", [])[:limit]:
        info = path["info"]
        legs, walk_min = [], 0
        for sp in path.get("subPath", []):
            kind = TRAFFIC.get(sp.get("trafficType"), "walk")
            minutes = int(sp.get("sectionTime") or 0)
            if kind == "walk":
                walk_min += minutes
                if minutes:
                    legs.append(Leg("walk", minutes, int(sp.get("distance") or 0)))
                continue
            lane = (sp.get("lane") or [{}])[0]
            legs.append(Leg(
                kind=kind, minutes=minutes, distance_m=int(sp.get("distance") or 0),
                name=str(lane.get("busNo") or lane.get("name") or ""),
                start=sp.get("startName", ""), end=sp.get("endName", ""),
                start_kor=sp.get("startNameKor", ""), stops=int(sp.get("stationCount") or 0),
                interval_min=int(sp["intervalTime"]) if sp.get("intervalTime") else None,
                seoul_route_id=str(lane["busLocalBlID"]) if lane.get("busLocalBlID") else None,
                seoul_stop_id=(str(sp["startLocalStationID"])
                               if sp.get("startLocalStationID") else None),
                start_xy=_xy(sp.get("startY"), sp.get("startX")),
                end_xy=_xy(sp.get("endY"), sp.get("endX")),
            ))
        routes.append(TransitRoute(
            total_min=int(info.get("totalTime") or 0), fare_krw=int(info.get("payment") or 0),
            bus_transfers=int(info.get("busTransitCount") or 0),
            subway_transfers=int(info.get("subwayTransitCount") or 0),
            walk_min=walk_min, legs=legs))
    return routes


def transit_routes(key: str, start: tuple[float, float], end: tuple[float, float],
                   limit: int = 5, timeout: float = 20) -> list[TransitRoute]:
    """Public-transit options from start to end ((lat, lon) tuples), in English."""
    params = urllib.parse.urlencode({
        "SX": start[1], "SY": start[0], "EX": end[1], "EY": end[0],
        "lang": 1, "SearchType": 0, "apiKey": key})
    req = urllib.request.Request(f"{ODSAY_URL}?{params}",
                                 headers={"Referer": APP_URL + "/", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_odsay(payload, limit)


def osrm_eta(mode: str, start: tuple[float, float], end: tuple[float, float],
             timeout: float = 15) -> tuple[float, float] | None:
    """(minutes, km) for car / walk / bike, or None if the service is unavailable."""
    url = OSRM_URL.format(profile=OSRM_PROFILES[mode],
                          a=f"{start[1]},{start[0]}", b=f"{end[1]},{end[0]}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        route = data["routes"][0]
        return round(route["duration"] / 60, 1), round(route["distance"] / 1000, 1)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def format_duration(minutes: float) -> str:
    minutes = int(round(minutes))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"
