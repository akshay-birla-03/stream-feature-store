"""Point-in-time correctness: the headline leakage-safety guarantees.

These tests construct event + label sets where a naive "join latest features
by entity" would leak future data, and assert that ``get_historical_features``
returns only as-of-time (strictly-before) values.
"""

import numpy as np
import pandas as pd

from featurestore import (
    FeatureSpec,
    FeatureStore,
    SlidingWindow,
    generate_events,
    generate_labels,
)


def cumulative_store():
    # A huge sliding window == cumulative count/sum over all history.
    specs = [
        FeatureSpec("txn_count", "count", SlidingWindow(1e12)),
        FeatureSpec("amount_sum", "sum", SlidingWindow(1e12), field="amount"),
    ]
    return FeatureStore(specs)


def test_strictly_before_excludes_event_at_label_time():
    """An event exactly AT the label timestamp must NOT be visible."""
    events = pd.DataFrame(
        {
            "entity_id": ["u", "u", "u"],
            "event_timestamp": [10.0, 20.0, 30.0],
            "amount": [1.0, 1.0, 1.0],
        }
    )
    store = cumulative_store().ingest(events)
    labels = pd.DataFrame({"entity_id": ["u"], "event_timestamp": [20.0]})
    out = store.get_historical_features(labels)
    # Only the event at t=10 is strictly before t=20; the t=20 event is excluded.
    assert out.loc[0, "txn_count"] == 1
    assert out.loc[0, "amount_sum"] == 1.0


def test_naive_join_would_leak_but_pit_does_not():
    """The naive 'latest value' join leaks; the PIT join does not."""
    events = pd.DataFrame(
        {
            "entity_id": ["u"] * 5,
            "event_timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "amount": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    store = cumulative_store().ingest(events)

    # Label at t=3.0: only events at t=1,2 are known (sum=20, count=2).
    labels = pd.DataFrame({"entity_id": ["u"], "event_timestamp": [3.0]})
    out = store.get_historical_features(labels)

    naive_latest = store.get_online_features("u")  # sees ALL 5 events (leak!)
    assert naive_latest["amount_sum"] == 50.0
    assert naive_latest["txn_count"] == 5

    assert out.loc[0, "txn_count"] == 2
    assert out.loc[0, "amount_sum"] == 20.0
    # Prove they differ: the PIT value is strictly less -> no leakage.
    assert out.loc[0, "amount_sum"] < naive_latest["amount_sum"]


def test_label_before_any_event_is_nan():
    events = pd.DataFrame(
        {"entity_id": ["u"], "event_timestamp": [100.0], "amount": [5.0]}
    )
    store = cumulative_store().ingest(events)
    labels = pd.DataFrame({"entity_id": ["u"], "event_timestamp": [50.0]})
    out = store.get_historical_features(labels)
    assert pd.isna(out.loc[0, "txn_count"])
    assert pd.isna(out.loc[0, "amount_sum"])


def test_sliding_window_expiry_reflected_in_pit():
    """A 10s sliding window: features expire; PIT reflects the value at query."""
    specs = [FeatureSpec("cnt_10s", "count", SlidingWindow(10.0))]
    events = pd.DataFrame(
        {
            "entity_id": ["u", "u", "u"],
            "event_timestamp": [0.0, 5.0, 100.0],
            "amount": [1.0, 1.0, 1.0],
        }
    )
    store = FeatureStore(specs).ingest(events)
    # Query at t=8: events at 0 and 5 are in the last-10s window -> count 2.
    out = store.get_historical_features(
        pd.DataFrame({"entity_id": ["u"], "event_timestamp": [8.0]})
    )
    assert out.loc[0, "cnt_10s"] == 2
    # Query at t=101: last snapshot was at t=100 where window (90,100] holds
    # only the t=100 event -> count 1.
    out2 = store.get_historical_features(
        pd.DataFrame({"entity_id": ["u"], "event_timestamp": [101.0]})
    )
    assert out2.loc[0, "cnt_10s"] == 1


def test_multiple_entities_no_crosstalk_in_pit():
    events = pd.DataFrame(
        {
            "entity_id": ["a", "b", "a", "b"],
            "event_timestamp": [1.0, 1.0, 2.0, 2.0],
            "amount": [100.0, 1.0, 100.0, 1.0],
        }
    )
    store = cumulative_store().ingest(events)
    labels = pd.DataFrame(
        {"entity_id": ["a", "b"], "event_timestamp": [3.0, 3.0]}
    )
    out = store.get_historical_features(labels).set_index("entity_id")
    assert out.loc["a", "amount_sum"] == 200.0
    assert out.loc["b", "amount_sum"] == 2.0


def test_pit_equals_bruteforce_recompute_on_random_stream():
    """End-to-end oracle: PIT count/sum must equal a brute-force recompute
    over exactly the events strictly before each label time."""
    events = generate_events(n_users=4, n_events=150, seed=3)
    labels = generate_labels(events, n_labels=25, seed=9)
    store = cumulative_store().ingest(events)
    out = store.get_historical_features(labels)

    ev = events.sort_values("event_timestamp")
    for i, row in out.iterrows():
        ent = row["entity_id"]
        t = labels.loc[i, "event_timestamp"]
        past = ev[(ev["entity_id"] == ent) & (ev["event_timestamp"] < t)]
        if past.empty:
            assert pd.isna(row["txn_count"])
            assert pd.isna(row["amount_sum"])
        else:
            assert row["txn_count"] == len(past)
            assert np.isclose(row["amount_sum"], past["amount"].sum())


def test_row_order_and_labels_preserved():
    events = pd.DataFrame(
        {"entity_id": ["u", "u"], "event_timestamp": [1.0, 2.0], "amount": [1.0, 1.0]}
    )
    store = cumulative_store().ingest(events)
    labels = pd.DataFrame(
        {
            "entity_id": ["u", "u", "u"],
            "event_timestamp": [3.0, 1.5, 2.5],
            "label": [1, 0, 1],
        }
    )
    out = store.get_historical_features(labels)
    # original columns + order preserved
    assert list(out["label"]) == [1, 0, 1]
    assert list(out["event_timestamp"]) == [3.0, 1.5, 2.5]
    assert "txn_count" in out.columns
