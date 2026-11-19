"""Tests for directions parsing (real ODsay fixture) and OSRM handling (no network)."""
import io
import json
from pathlib import Path

import pytest

import src.directions as d

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "odsay_coex_sinseoldong.json")
                     .read_text(encoding="utf-8"))


def test_parse_real_odsay_response():
    routes = d.parse_odsay(FIXTURE)
    assert len(routes) == 3
    best = routes[0]
    assert best.total_min > 0 and best.fare_krw > 0
    assert best.legs and all(leg.kind in {"walk", "bus", "subway"} for leg in best.legs)
    # English output (lang=1): ASCII station names on transit legs
    transit = [leg for leg in best.legs if leg.kind != "walk"]
    assert all(leg.start.isascii() and leg.end.isascii() for leg in transit)


def test_bus_legs_carry_seoul_ids_for_live_status():
    bus_legs = [leg for r in d.parse_odsay(FIXTURE) for leg in r.legs if leg.kind == "bus"]
    assert bus_legs, "fixture should contain bus legs"
    for leg in bus_legs:
        assert leg.seoul_route_id and leg.seoul_route_id.isdigit()
        assert leg.seoul_stop_id and leg.seoul_stop_id.isdigit()
        assert leg.name and leg.stops > 0 and leg.start_xy


def test_transfers_and_modes():
    r = d.TransitRoute(40, 1650, bus_transfers=1, subway_transfers=1, walk_min=8,
                       legs=[d.Leg("bus", 10), d.Leg("subway", 20)])
    assert r.transfers == 1 and r.modes == "Bus + Subway"


@pytest.mark.parametrize("payload, code", [
    ({"error": {"msg": "출, 도착지가 700m이내입니다.", "code": "-98"}}, "-98"),
    ({"error": [{"code": "500", "message": "[ApiKeyAuthFailed] ApiKey authentication failed."}]},
     "500"),
])
def test_odsay_errors_raise(payload, code):
    with pytest.raises(d.DirectionsError, match=code):
        d.parse_odsay(payload)


def test_transit_routes_sends_referer(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["referer"] = req.get_header("Referer")
        seen["url"] = req.full_url
        return io.BytesIO(json.dumps(FIXTURE).encode())

    monkeypatch.setattr(d.urllib.request, "urlopen", fake_urlopen)
    routes = d.transit_routes("KEY", (37.5119, 127.0591), (37.5745, 127.0254))
    assert routes and seen["referer"].startswith(d.APP_URL)
    assert "lang=1" in seen["url"] and "SX=127.0591" in seen["url"]


def test_osrm_eta_parses_and_fails_gracefully(monkeypatch):
    ok = {"routes": [{"duration": 696.0, "distance": 9600.0}]}
    monkeypatch.setattr(d.urllib.request, "urlopen",
                        lambda req, timeout=0: io.BytesIO(json.dumps(ok).encode()))
    assert d.osrm_eta("car", (37.5, 127.0), (37.6, 127.1)) == (11.6, 9.6)

    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(d.urllib.request, "urlopen", boom)
    assert d.osrm_eta("walk", (37.5, 127.0), (37.6, 127.1)) is None


def test_format_duration():
    assert d.format_duration(34) == "34 min"
    assert d.format_duration(125.4) == "2 h 05 min"
