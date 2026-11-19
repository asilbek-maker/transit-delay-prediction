"""Build data/reference/bus_stops.json: every stop on the tracked bus routes.

The Streamlit app searches this file offline, so looking up a location costs no
API calls; only the live arrival request for the chosen stop does.

Usage:  python scripts/build_stop_index.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.collectors.seoul_bus import KST, ROUTES, fetch_route, load_key  # noqa: E402
from scripts.xlsx_lite import read_rows  # noqa: E402

OUT = ROOT / "data" / "reference" / "bus_stops.json"
# 서울시 버스정류소 위치정보 (Seoul Open Data Plaza OA-15067), WGS84 lon/lat.
LOCATIONS_URL = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
LOCATIONS_FORM = {"infId": "OA-15067", "infSeq": "1", "seq": "58"}   # 2026-09-02 file


def stop_coordinates() -> dict[str, tuple[float, float]]:
    """NODE_ID (= stId) -> (lat, lon) for every Seoul bus stop."""
    data = "&".join(f"{k}={v}" for k, v in LOCATIONS_FORM.items()).encode()
    req = urllib.request.Request(LOCATIONS_URL, data=data, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://data.seoul.go.kr/dataList/OA-15067/S/1/datasetView.do"})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stops.xlsx"
        path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        rows = read_rows(path)
    header = rows[0]
    i_id, i_x, i_y = header.index("NODE_ID"), header.index("X좌표"), header.index("Y좌표")
    coords = {}
    for r in rows[1:]:
        try:
            coords[r[i_id]] = (round(float(r[i_y]), 6), round(float(r[i_x]), 6))
        except (ValueError, IndexError):
            continue
    return coords


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    key = load_key()
    if not key:
        print("DATA_GO_KR_KEY is not set")
        return 1
    coords = stop_coordinates()
    print(f"stop locations: {len(coords)}")
    routes = {}
    missing = 0
    for route_id, number in ROUTES.items():
        rows = fetch_route(key, route_id)
        if not rows:
            print(f"route {number}: no data, skipped")
            continue
        stops = []
        for r in rows:
            lat, lon = coords.get(r["stId"], (None, None))
            missing += lat is None
            stops.append({"ord": int(r["staOrd"]), "stId": r["stId"], "arsId": r["arsId"],
                          "stNm": r["stNm"], "lat": lat, "lon": lon})
        stops.sort(key=lambda s: s["ord"])
        routes[route_id] = {"rtNm": number, "term": int(rows[0].get("term") or 0),
                            "stops": stops}
        print(f"route {number}: {len(stops)} stops")
    print(f"stops without coordinates: {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"built_at": datetime.now(KST).isoformat(timespec="seconds"),
                               "routes": routes}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
