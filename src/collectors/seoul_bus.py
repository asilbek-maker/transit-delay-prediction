"""Collect live Seoul bus arrivals for whole routes (서울특별시_버스도착정보조회 서비스).

For every stop on each tracked route, the API reports the next two buses: vehicle ID,
the stop each bus is currently at, predicted seconds to arrival, crowding and the
scheduled headway. Repeated snapshots let us reconstruct when each vehicle actually
reached each stop, so we can measure real delays and bus bunching later.

Standard library only. Output is one gzip CSV per day (KST); gzip members can be
appended safely and pandas reads the result directly.

Usage:
    python -m src.collectors.seoul_bus --once
    python -m src.collectors.seoul_bus --minutes 345

The key is read from DATA_GO_KR_KEY (environment or .env).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
API = ("http://ws.bus.go.kr/api/rest/arrive/getArrInfoByRouteAll"
       "?serviceKey={key}&busRouteId={route}&resultType=json")

# busRouteId -> route number. All are 간선 (trunk) routes, verified against the live API.
ROUTES = {
    "100100025": "146",   # 135 stops, every ~7 min
    "100100031": "152",   # 131 stops, long cross-city trunk
    "100100051": "301",   # 100 stops, every ~7 min
    "100100052": "302",   # 126 stops, every ~12 min (Seongnam 상대원 → 왕십리)
    "100100053": "303",   # 106 stops, every ~9 min (Seongnam 상대원 → 신설동)
    "100100061": "370",   # every ~5 min, most frequent
    "100100118": "753",   # 104 stops, every ~12 min
}
FIELDS = [
    "polled_at_kst", "mkTm", "busRouteId", "rtNm", "routeType", "term", "staOrd",
    "stId", "stNm", "arsId",
    "vehId1", "plainNo1", "sectOrd1", "traTime1", "traSpd1", "isArrive1", "isLast1",
    "full1", "rerdie_Div1", "reride_Num1", "busType1",
    "vehId2", "plainNo2", "sectOrd2", "traTime2", "isArrive2", "isLast2",
    "full2", "rerdie_Div2", "reride_Num2",
    "arrmsg1", "arrmsg2", "deTourAt",
]
# Buses run roughly 04:00-01:00 KST; skipping the night saves quota.
SERVICE_START_HOUR, SERVICE_END_HOUR = 5, 1


class QuotaExceeded(RuntimeError):
    """The daily request limit for this key has been reached."""


def load_key() -> str:
    key = os.environ.get("DATA_GO_KR_KEY", "").strip()
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATA_GO_KR_KEY="):
                key = line.split("=", 1)[1].strip()
    return key


def in_service(now: datetime) -> bool:
    return not (SERVICE_END_HOUR <= now.hour < SERVICE_START_HOUR)


def parse_response(body: str) -> list[dict]:
    """Return the stop list, or raise QuotaExceeded / ValueError on API errors."""
    if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in body:
        raise QuotaExceeded(body[:200])
    payload = json.loads(body)
    if "msgHeader" not in payload:                       # e.g. 401 Unauthorized JSON
        raise ValueError(payload.get("message") or body[:200])
    header = payload["msgHeader"]
    if str(header.get("headerCd")) not in ("0", "4"):     # 4 = no results (night)
        if "초과" in str(header.get("headerMsg")):
            raise QuotaExceeded(header.get("headerMsg"))
        raise ValueError(f"API {header.get('headerCd')}: {header.get('headerMsg')}")
    return (payload.get("msgBody") or {}).get("itemList") or []


def fetch_route(key: str, route_id: str, retries: int = 3) -> list[dict]:
    url = API.format(key=urllib.parse.quote(key, safe=""), route=route_id)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                body = resp.read().decode("utf-8", "replace")
            return parse_response(body)
        except QuotaExceeded:
            raise
        except urllib.error.HTTPError as exc:                # 401/403 come back as HTTP errors
            body = exc.read().decode("utf-8", "replace")
            print(f"[{ROUTES.get(route_id, route_id)}] HTTP {exc.code}: {body[:150]}",
                  file=sys.stderr)
            return []
        except (OSError, ValueError) as exc:
            print(f"[{ROUTES.get(route_id, route_id)}] attempt {attempt}: {exc}", file=sys.stderr)
            if attempt == retries:
                return []
            time.sleep(5 * attempt)
    return []


def append_rows(out_dir: Path, rows: list[dict], polled_at: datetime) -> Path:
    """Append rows as a new gzip member to out_dir/arrivals_YYYY-MM-DD.csv.gz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"arrivals_{polled_at:%Y-%m-%d}.csv.gz"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    if not path.exists():
        writer.writeheader()
    stamp = polled_at.isoformat(timespec="seconds")
    for row in rows:
        writer.writerow({**row, "polled_at_kst": stamp})
    with gzip.open(path, "at", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    return path


def poll_once(key: str, routes: list[str], out_dir: Path) -> int:
    now = datetime.now(KST)
    total = 0
    for route_id in routes:
        rows = fetch_route(key, route_id)
        if rows:
            path = append_rows(out_dir, rows, now)
            buses = len({r.get("vehId1") for r in rows if r.get("vehId1") not in (None, "0")})
            total += len(rows)
            print(f"{now:%H:%M:%S} route {ROUTES.get(route_id, route_id):>4}: "
                  f"{len(rows):3d} stops, {buses:2d} buses → {path.name}")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--routes", nargs="+", default=list(ROUTES), help="busRouteIds")
    parser.add_argument("--out", type=Path, default=Path("data/live/bus"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--minutes", type=float, default=0)
    # 7 routes x one call per poll every 10 min over ~20 service hours ≈ 840 calls/day.
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--require-key", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    key = load_key()
    if not key:
        print("No DATA_GO_KR_KEY configured" + ("; skipping." if args.require_key else "."))
        return 0 if args.require_key else 1
    print("Routes:", ", ".join(ROUTES.get(r, r) for r in args.routes))

    try:
        if args.once or args.minutes <= 0:
            return 0 if poll_once(key, args.routes, args.out) else 1
        deadline = time.monotonic() + args.minutes * 60
        polls = rows = 0
        while time.monotonic() < deadline:
            started = time.monotonic()
            if in_service(datetime.now(KST)):
                rows += poll_once(key, args.routes, args.out)
                polls += 1
            # never sleep past the deadline (a run should end on time)
        time.sleep(max(0, min(args.interval - (time.monotonic() - started),
                              deadline - time.monotonic())))
        print(f"Done: {polls} polls, {rows} rows")
    except QuotaExceeded as exc:
        print(f"Daily quota reached, stopping until tomorrow: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
