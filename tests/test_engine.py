import pandas as pd
import pytest

from featurestore.engine import FeatureSpec, StreamEngine
from featurestore.windows import SlidingWindow, TumblingWindow


def make_engine():
    specs = [
        FeatureSpec("cnt_10s", "count", SlidingWindow(10.0)),
        FeatureSpec("sum_10s", "sum", SlidingWindow(10.0), field="amount"),
        FeatureSpec("cnt_bucket", "count", TumblingWindow(10.0)),
    ]
    return StreamEngine(specs)


def test_sliding_eviction_in_engine():
    eng = make_engine()
    # events at t=0,1,2 then t=11 which evicts t=0 (start=1, evict t<=1)
    for t, amt in [(0, 5), (1, 5), (2, 5), (11, 5)]:
        eng.ingest_event("u", t, {"amount": amt})
    state = eng.get_online_state("u")
    # window (1, 11]: events at t=2 and t=11 -> count 2, sum 10
    assert state["cnt_10s"] == 2
    assert state["sum_10s"] == 10.0


def test_tumbling_resets_across_buckets():
    eng = make_engine()
    for t in [0, 3, 9]:  # bucket 0
        eng.ingest_event("u", t, {"amount": 1})
    assert eng.get_online_state("u")["cnt_bucket"] == 3
    eng.ingest_event("u", 12, {"amount": 1})  # bucket 1 -> reset
    assert eng.get_online_state("u")["cnt_bucket"] == 1


def test_per_entity_isolation():
    eng = make_engine()
    eng.ingest_event("a", 0, {"amount": 10})
    eng.ingest_event("b", 1, {"amount": 20})
    assert eng.get_online_state("a")["sum_10s"] == 10.0
    assert eng.get_online_state("b")["sum_10s"] == 20.0
    assert set(eng.entities()) == {"a", "b"}


def test_out_of_order_rejected():
    eng = make_engine()
    eng.ingest_event("u", 5, {"amount": 1})
    with pytest.raises(ValueError):
        eng.ingest_event("u", 4, {"amount": 1})


def test_unknown_entity_online_state_none():
    eng = make_engine()
    assert eng.get_online_state("ghost") is None


def test_duplicate_feature_names_rejected():
    with pytest.raises(ValueError):
        StreamEngine(
            [
                FeatureSpec("x", "count", SlidingWindow(1.0)),
                FeatureSpec("x", "sum", SlidingWindow(1.0)),
            ]
        )


def test_empty_specs_rejected():
    with pytest.raises(ValueError):
        StreamEngine([])


def test_ingest_frame_matches_event_by_event():
    df = pd.DataFrame(
        {
            "entity_id": ["u", "u", "u"],
            "event_timestamp": [0.0, 1.0, 2.0],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    eng = make_engine().ingest_frame(df)
    assert eng.n_events == 3
    assert eng.get_online_state("u")["sum_10s"] == 6.0


def test_history_is_monotonic_and_recorded_per_event():
    eng = make_engine()
    for t in [0, 1, 2]:
        eng.ingest_event("u", t, {"amount": 1})
    hist = eng._history["u"]
    assert hist.times == [0.0, 1.0, 2.0]
    assert len(hist.rows) == 3
