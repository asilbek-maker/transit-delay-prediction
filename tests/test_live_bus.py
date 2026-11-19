"""Tests for the live stop lookup (offline: uses the committed stop index)."""
from src.live_bus import load_stop_index, search_stops, summarize_stop

INDEX = load_stop_index()


def test_index_covers_all_tracked_routes():
    assert {r["rtNm"] for r in INDEX["routes"].values()} >= {"146", "152", "301", "302", "303",
                                                              "370", "753"}
    assert all(r["term"] > 0 and len(r["stops"]) > 50 for r in INDEX["routes"].values())


def test_search_groups_routes_by_physical_stop():
    results = search_stops(INDEX, "신설동역.서울풍물시장")
    stop = next(s for s in results if s["arsId"] == "06008")
    assert {r["rtNm"] for r in stop["routes"]} == {"303", "370"}
    assert all(r["next_stop"] for r in stop["routes"])        # direction hint present


def test_search_ignores_spaces_and_dots_and_matches_stop_number():
    assert search_stops(INDEX, "신설동역 서울풍물시장")
    assert search_stops(INDEX, "06008")[0]["arsId"] == "06008"
    assert search_stops(INDEX, "") == []
    assert search_stops(INDEX, "존재하지않는정류장") == []


def _row(t1, t2=None, **extra):
    return {"rtNm": "146", "traTime1": str(t1), "traTime2": str(t2 or 0),
            "reride_Num1": "3", "arrmsg1": "", **extra}


def test_on_schedule_when_wait_within_interval():
    s = summarize_stop(_row(5 * 60, 14 * 60), term=7)
    assert (s.status, s.delay_min, s.next_min) == ("On schedule", 0.0, 5.0)


def test_delayed_when_wait_exceeds_interval_plus_tolerance():
    s = summarize_stop(_row(12.7 * 60, 19 * 60), term=7)
    assert s.status == "Delayed" and s.delay_min == 5.7


def test_bunched_when_buses_arrive_close_together():
    s = summarize_stop(_row(4 * 60, 5 * 60), term=10)
    assert s.status == "Bunched"


def test_no_bus_uses_api_message():
    s = summarize_stop(_row(0, 0, arrmsg1="운행종료"), term=7)
    assert s.next_min is None and s.status == "운행종료"


def test_arriving_now():
    s = summarize_stop(_row(30, 600, isArrive1="1"), term=7)
    assert s.next_min == 0.0 and s.status == "On schedule"
