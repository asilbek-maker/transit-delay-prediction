"""Build data/reference/bus_stops.json: every stop on the tracked bus routes.

The Streamlit app searches this file offline, so looking up a location costs no
API calls; only the live arrival request for the chosen stop does.

Usage:  python scripts/build_stop_index.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.collectors.seoul_bus import KST, ROUTES, fetch_route, load_key  # noqa: E402

OUT = ROOT / "data" / "reference" / "bus_stops.json"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    key = load_key()
    if not key:
        print("DATA_GO_KR_KEY is not set")
        return 1
    routes = {}
    for route_id, number in ROUTES.items():
        rows = fetch_route(key, route_id)
        if not rows:
            print(f"route {number}: no data, skipped")
            continue
        stops = [{"ord": int(r["staOrd"]), "stId": r["stId"], "arsId": r["arsId"],
                  "stNm": r["stNm"]} for r in rows]
        stops.sort(key=lambda s: s["ord"])
        routes[route_id] = {"rtNm": number, "term": int(rows[0].get("term") or 0),
                            "stops": stops}
        print(f"route {number}: {len(stops)} stops")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"built_at": datetime.now(KST).isoformat(timespec="seconds"),
                               "routes": routes}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
