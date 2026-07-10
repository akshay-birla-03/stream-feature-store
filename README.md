# stream-feature-store

A lightweight **streaming feature engineering + feature store** library for
real-time ML. It ingests an event stream, maintains **windowed aggregate
features** (count / sum / mean / max / min / distinct-count over sliding and
tumbling time windows) **per entity, incrementally**, and serves
**point-in-time-correct** feature lookups for both online serving and offline
training-set generation.

Pure Python (numpy / pandas). Fully offline: event streams are synthetic and
generated in code. No external services, no network.

---

## Why point-in-time correctness matters

This is the crux of the whole library, so it is worth stating plainly.

When you build a training set for a model, each label row has a timestamp: the
moment the prediction *would have been made* in production. To train without
cheating, every feature attached to that row must be computed using **only data
that existed strictly before that moment**.

The naive approach — "join the current feature value onto each label by
entity" — silently attaches the feature's value *as it is now*, which for a
historical label includes information from **after** the label was created.
That is **label leakage**. Two failure modes follow:

1. **Offline/online skew.** The model trains on feature values that serving can
   never reproduce (serving does not have the future). Offline metrics look
   great; production performance collapses.
2. **Target leakage.** If future events correlate with the label, the model
   learns to "peek", inflating validation scores and hiding a model that has
   learned nothing generalizable.

`get_historical_features` performs a **point-in-time (as-of) join**: for each
`(entity_id, event_timestamp)` it returns the feature vector exactly as it was
known *strictly before* `event_timestamp`. The strict inequality is the core
guard — an event landing exactly at the label time is treated as not-yet-known.

---

## Worked example of the point-in-time join

Five events for user `u`, each `amount=10`, at t = 1, 2, 3, 4, 5. We ask for a
label at **t = 3.0** with a cumulative count/sum feature:

```python
import pandas as pd
from featurestore import FeatureStore, FeatureSpec, SlidingWindow

specs = [
    FeatureSpec("txn_count", "count", SlidingWindow(1e12)),
    FeatureSpec("amount_sum", "sum", SlidingWindow(1e12), field="amount"),
]
events = pd.DataFrame({
    "entity_id": ["u"] * 5,
    "event_timestamp": [1, 2, 3, 4, 5],
    "amount": [10, 10, 10, 10, 10],
})
store = FeatureStore(specs).ingest(events)

labels = pd.DataFrame({"entity_id": ["u"], "event_timestamp": [3.0], "label": [1]})
print(store.get_historical_features(labels))
```

Output:

```
entity_id  event_timestamp  label  txn_count  amount_sum
        u              3.0      1          2        20.0
```

At t = 3.0 only the events at t = 1 and t = 2 are known, so `txn_count = 2` and
`amount_sum = 20.0`. Note the event **at** t = 3 is *excluded* (strictly
before). A naive join would instead attach the online value —
`txn_count = 5, amount_sum = 50.0` — leaking all future events into the row.

---

## Architecture

```
                        synthetic events (entity, ts, amount, merchant)
                                        |
                                        v
   +-------------------------------------------------------------------+
   |                          StreamEngine                             |
   |   ingest events in non-decreasing time order, per entity:         |
   |                                                                   |
   |   +-- SlidingWindow --+   evict aged events (t <= now-duration),  |
   |   |  FIFO (t, value)  |   then add new -> incremental aggregators  |
   |   +-------------------+                                            |
   |   +-- TumblingWindow -+   reset aggregator when bucket changes     |
   |   +-------------------+                                            |
   |                                                                   |
   |   aggregators: Count Sum Mean Max Min DistinctCount (add/evict)   |
   |                                                                   |
   |   after EACH event -> append snapshot to per-entity history       |
   |      history[entity] = [(t0, {...}), (t1, {...}), ...]  (monotone) |
   +-------------------------------------------------------------------+
              |                                          |
              v                                          v
     get_online_features(entity)            get_historical_features(labels)
     latest snapshot (O(1))                 as-of join: bisect history for
     -> online serving / API                the last snapshot with t < label_t
                                            -> leakage-safe training set
```

Two properties make the point-in-time join correct and cheap:

* **Snapshots are recorded after each event.** A snapshot stamped `t` reflects
  exactly the events with timestamp `<= t` and nothing from the future.
* **History times are monotonic**, so an as-of lookup is a binary search
  (`bisect_left(times, label_t) - 1`) — the newest snapshot strictly before the
  label time.

---

## Installation

```bash
pip install -e ".[dev]"     # editable install with dev extras
```

Only depends on numpy, pandas, scikit-learn, fastapi, uvicorn, joblib.

---

## Usage

### Offline: leakage-safe training-set generation

```python
from featurestore import (
    FeatureStore, default_feature_specs, generate_events, generate_labels,
)

events = generate_events(n_users=5, n_events=300, seed=7)
store = FeatureStore(default_feature_specs()).ingest(events)

labels = generate_labels(events, n_labels=8, seed=8)
training_set = store.get_historical_features(labels)   # point-in-time join
```

Each feature column in `training_set` holds the value known strictly before that
row's `event_timestamp`. Feed it straight into scikit-learn.

### Online: low-latency serving

```python
store.get_online_features("user_1")
# {'txn_count_5m': 3, 'amount_sum_5m': 262.35, 'amount_mean_1h': 98.71, ...}
```

### Defining features

```python
from featurestore import FeatureSpec, SlidingWindow, TumblingWindow

specs = [
    FeatureSpec("txn_count_5m", "count", SlidingWindow(300.0)),
    FeatureSpec("amount_sum_5m", "sum", SlidingWindow(300.0), field="amount"),
    FeatureSpec("amount_mean_1h", "mean", SlidingWindow(3600.0), field="amount"),
    FeatureSpec("txn_count_hour", "count", TumblingWindow(3600.0)),
    FeatureSpec("distinct_merchants_1h", "distinct_count",
                SlidingWindow(3600.0), field="merchant"),
]
```

* **SlidingWindow(d)** — trailing window, membership `now - d < t <= now`
  (leading edge inclusive, trailing edge exclusive; deterministic eviction).
* **TumblingWindow(d)** — non-overlapping buckets aligned to an origin; the
  aggregator resets when an event crosses into a new bucket.

### CLI demo

```bash
featurestore --n-users 5 --n-events 300 --n-labels 8
```

Prints online features for a few entities and a small point-in-time training
set.

### Online API (FastAPI)

```bash
uvicorn featurestore.api:app --reload
# GET /health
# GET /features/{entity_id}
```

Or via Docker:

```bash
docker build -t stream-feature-store .
docker run -p 8000:8000 stream-feature-store
```

---

## Design of the incremental aggregation

Recomputing an aggregate over an entire window on every event is O(window) per
tick. Instead, each aggregator supports `add(value)` / `evict(value)` so an
update is amortised O(1):

| Aggregator      | State                    | add / evict |
| --------------- | ------------------------ | ----------- |
| Count           | integer counter          | +/- 1       |
| Sum / Mean      | running total (+ count)  | +/- value   |
| Max / Min       | sorted multiset          | insert / remove any element |
| DistinctCount   | `Counter` multiset       | ref-count per value |

`Max`/`Min` keep a **sorted multiset** rather than a single extreme, because in
a sliding window the event that expires may not be the current extreme — a
naive "track the max" would be wrong after the max ages out. `DistinctCount`
ref-counts values so evicting one occurrence of a repeated value does not
prematurely drop it from the distinct set.

---

## Testing

```bash
pytest -q
```

47 tests covering:

* **Window membership** — half-open sliding edges, tumbling bucket alignment.
* **Aggregator correctness** — every aggregator checked incrementally against a
  brute-force recompute, including a simulated sliding window that adds the
  newest and evicts the oldest at each step.
* **Engine** — sliding eviction, tumbling reset across buckets, per-entity
  isolation, out-of-order rejection, monotonic history.
* **Point-in-time correctness (no leakage)** — the headline suite. It includes
  a test that ingests five events and asserts the as-of value at a mid-stream
  label time is *strictly less* than the online (leaky) value, plus a
  random-stream oracle test that checks the point-in-time join equals a
  brute-force recompute over exactly the events strictly before each label
  timestamp.
* **API** — `/health` and `/features/{id}` via FastAPI `TestClient`, including a
  404 for unknown entities.

---

## License

MIT (c) Akshay GC
