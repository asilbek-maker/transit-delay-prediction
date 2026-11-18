"""Collect live Seoul subway train positions (서울시 지하철 실시간 열차 위치정보).

Polls the Seoul real-time subway API for the configured lines and appends every
train's position to a daily CSV. Repeated snapshots let us reconstruct when each
train reached each station, from which run times and delays are derived later.

Standard library only, so it runs anywhere (laptop, GitHub Actions) without pip.

Usage:
    python -m src.collectors.seoul_subway --once
    python -m src.collectors.seoul_subway --minutes 350 --interval 180

The API key is read from SEOUL_SUBWAY_KEY (environment or .env). Without it the
public "sample" key is used, which only returns 5 trains per call (testing only).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
API = "http://swopenapi.seoul.go.kr/api/subway/{key}/json/realtimePosition/0/{rows}/{line}"
DEFAULT_LINES = ["2호선", "9호선"]
FIELDS = [
    "polled_at_kst", "subwayId", "subwayNm", "statnId", "statnNm", "trainNo",
    "recptnDt", "updnLine", "statnTid", "statnTnm", "trainSttus", "directAt", "lstcarAt",
]
# Service runs ~05:30-01:00 KST. Skipping the night keeps us under the daily quota.
SERVICE_START_HOUR, SERVICE_END_HOUR = 5, 1


def load_key() -> str:
    """API key from the environment, then from a local .env file, else 'sample'."""
    key = os.environ.get("SEOUL_SUBWAY_KEY", "").strip()
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("SEOUL_SUBWAY_KEY="):
                key = line.split("=", 1)[1].strip()
    return key or "sample"


def in_service(now: datetime) -> bool:
    return not (SERVICE_END_HOUR <= now.hour < SERVICE_START_HOUR)


def fetch_line(key: str, line: str, retries: int = 3) -> list[dict]:
    """Return all train positions on one line (empty list on API-level errors)."""
    rows = 5 if key == "sample" else 200
    url = API.format(key=urllib.parse.quote(key), rows=rows, line=urllib.parse.quote(line))
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except (OSError, ValueError) as exc:          # network error or bad JSON
            print(f"[{line}] attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt == retries:
                return []
            time.sleep(5 * attempt)
    status = payload.get("errorMessage") or {}
    code = status.get("code") or payload.get("code", "")
    if code != "INFO-000":
        # e.g. INFO-200 (no trains, normal at night) or ERROR-337 (daily quota hit)
        print(f"[{line}] API {code}: {status.get('message') or payload.get('message')}",
              file=sys.stderr)
        return []
    return payload.get("realtimePositionList", [])


def append_rows(out_dir: Path, rows: list[dict], polled_at: datetime) -> Path:
    """Append rows to out_dir/positions_YYYY-MM-DD.csv (KST date)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"positions_{polled_at:%Y-%m-%d}.csv"
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        stamp = polled_at.isoformat(timespec="seconds")
        for row in rows:
            writer.writerow({**row, "polled_at_kst": stamp})
    return path


def poll_once(key: str, lines: list[str], out_dir: Path) -> int:
    now = datetime.now(KST)
    total = 0
    for line in lines:
        rows = fetch_line(key, line)
        if rows:
            path = append_rows(out_dir, rows, now)
            total += len(rows)
            print(f"{now:%H:%M:%S} {line}: {len(rows):3d} trains → {path.name}")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lines", nargs="+", default=DEFAULT_LINES)
    parser.add_argument("--out", type=Path, default=Path("data/live/subway"))
    parser.add_argument("--once", action="store_true", help="poll a single time")
    parser.add_argument("--minutes", type=float, default=0, help="keep polling this long")
    parser.add_argument("--interval", type=int, default=180, help="seconds between polls")
    parser.add_argument("--require-key", action="store_true",
                        help="exit quietly if only the sample key is available")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Korean line names on Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    key = load_key()
    print(f"Key: {'sample (5 trains/call, testing only)' if key == 'sample' else 'personal'}"
          f" | lines: {', '.join(args.lines)}")
    if args.require_key and key == "sample":
        print("No SEOUL_SUBWAY_KEY configured; skipping scheduled collection.")
        return 0

    if args.once or args.minutes <= 0:
        return 0 if poll_once(key, args.lines, args.out) else 1

    deadline = time.monotonic() + args.minutes * 60
    polls = rows = 0
    while time.monotonic() < deadline:
        started = time.monotonic()
        if in_service(datetime.now(KST)):
            rows += poll_once(key, args.lines, args.out)
            polls += 1
        time.sleep(max(0, args.interval - (time.monotonic() - started)))
    print(f"Done: {polls} polls, {rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
