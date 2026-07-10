"""Streaming feature engine.

``StreamEngine`` consumes an ordered event stream and, for every configured
feature, maintains incremental windowed aggregate state **per entity**. After
each event it records a time-stamped snapshot of every feature's value into a
per-entity history. That history is what makes leakage-safe, point-in-time
feature reconstruction possible downstream (see :mod:`featurestore.store`).

Design notes
------------
* Events must be fed in non-decreasing timestamp order. The engine validates
  this and raises otherwise -- out-of-order ingestion would corrupt window
  eviction and the monotonic history assumption relied on by the point-in-time
  join.
* Sliding windows keep a per-feature FIFO of ``(t, value)`` pairs; on each new
  event we first evict everything that has aged past the trailing edge, then
  add the new value. This keeps every update amortised O(1) (+ O(log n) for
  Max/Min).
* Tumbling windows retain only the current bucket: when an event crosses into a
  new bucket the aggregator is reset before the new value is added.
* A snapshot is recorded *after* fully processing each event, so a snapshot at
  time ``t`` reflects exactly the events with timestamp ``<= t`` and nothing
  from the future.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .aggregations import make_aggregator
from .windows import SlidingWindow, TumblingWindow, to_epoch_seconds

__all__ = ["FeatureSpec", "StreamEngine"]


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative definition of one windowed feature.

    Parameters
    ----------
    name:
        Unique feature name (column name in outputs).
    aggregation:
        Registry key: one of ``count``/``sum``/``mean``/``max``/``min``/
        ``distinct_count``.
    window:
        A :class:`SlidingWindow` or :class:`TumblingWindow`.
    field:
        Event column supplying the value to aggregate. Ignored by ``count``
        (which counts events regardless of field).
    """

    name: str
    aggregation: str
    window: object
    field: str = "amount"


class _FeatureState:
    """Mutable per-(entity, feature) aggregation state."""

    __slots__ = ("spec", "agg", "buffer", "bucket_index")

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self.agg = make_aggregator(spec.aggregation)
        # FIFO of (epoch_seconds, value) for sliding eviction.
        self.buffer: deque = deque()
        self.bucket_index: int | None = None

    def update(self, t_epoch: float, value) -> None:
        window = self.spec.window
        if isinstance(window, SlidingWindow):
            start = t_epoch - window.duration
            buf = self.buffer
            while buf and buf[0][0] <= start:
                _, old_val = buf.popleft()
                self.agg.evict(old_val)
            self.agg.add(value)
            buf.append((t_epoch, value))
        elif isinstance(window, TumblingWindow):
            idx = window.bucket_index(t_epoch)
            if self.bucket_index is None or idx != self.bucket_index:
                # New bucket: discard everything from the previous bucket.
                self.agg = make_aggregator(self.spec.aggregation)
                self.bucket_index = idx
            self.agg.add(value)
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported window type: {type(window)!r}")

    @property
    def value(self):
        return self.agg.value


@dataclass
class _EntityHistory:
    """Time-ordered snapshots for a single entity."""

    times: list = field(default_factory=list)  # epoch seconds, non-decreasing
    rows: list = field(default_factory=list)  # list[dict[feature_name -> value]]


class StreamEngine:
    """Maintain windowed features over an ordered event stream."""

    def __init__(self, specs: list[FeatureSpec]) -> None:
        if not specs:
            raise ValueError("at least one FeatureSpec is required")
        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        self.specs = list(specs)
        self.feature_names = names
        # entity_id -> {feature_name -> _FeatureState}
        self._state: dict = {}
        # entity_id -> _EntityHistory
        self._history: dict = {}
        self._last_time: float | None = None
        self._n_events = 0

    # -- ingestion ---------------------------------------------------------
    def ingest_event(self, entity_id, timestamp, payload: dict) -> None:
        """Process a single event.

        ``payload`` maps event field names to values. Timestamps must be fed
        in non-decreasing order across the whole stream.
        """
        t = to_epoch_seconds(timestamp)
        if self._last_time is not None and t < self._last_time:
            raise ValueError(
                "events must arrive in non-decreasing timestamp order "
                f"(got {t} after {self._last_time})"
            )
        self._last_time = t

        states = self._state.get(entity_id)
        if states is None:
            states = {s.name: _FeatureState(s) for s in self.specs}
            self._state[entity_id] = states
            self._history[entity_id] = _EntityHistory()

        for _name, st in states.items():
            fld = st.spec.field
            if st.spec.aggregation == "count":
                value = 1  # counted regardless of field
            else:
                value = payload[fld]
            st.update(t, value)

        snapshot = {name: st.value for name, st in states.items()}
        hist = self._history[entity_id]
        hist.times.append(t)
        hist.rows.append(snapshot)
        self._n_events += 1

    def ingest_frame(
        self,
        events: pd.DataFrame,
        entity_col: str = "entity_id",
        time_col: str = "event_timestamp",
    ) -> StreamEngine:
        """Ingest a whole DataFrame of events (sorted by ``time_col``)."""
        ordered = events.sort_values(time_col, kind="stable")
        payload_cols = [c for c in ordered.columns if c not in (entity_col, time_col)]
        for row in ordered.itertuples(index=False):
            d = row._asdict()
            entity = d[entity_col]
            ts = d[time_col]
            payload = {c: d[c] for c in payload_cols}
            self.ingest_event(entity, ts, payload)
        return self

    # -- online serving ----------------------------------------------------
    def get_online_state(self, entity_id) -> dict | None:
        """Latest computed feature values for an entity (or ``None``)."""
        hist = self._history.get(entity_id)
        if hist is None or not hist.rows:
            return None
        return dict(hist.rows[-1])

    def entities(self) -> list:
        return list(self._state.keys())

    @property
    def n_events(self) -> int:
        return self._n_events

    # -- point-in-time reconstruction -------------------------------------
    def features_asof(self, entity_id, timestamp) -> dict | None:
        """Feature values as known **strictly before** ``timestamp``.

        Returns the most recent snapshot whose recorded time is ``< t``. If no
        such snapshot exists (entity unknown or all events at/after ``t``),
        returns ``None``. The strict inequality is the core leakage guard: an
        event occurring exactly at the label time is *not* visible.
        """
        hist = self._history.get(entity_id)
        if hist is None or not hist.times:
            return None
        t = to_epoch_seconds(timestamp)
        # First index with time >= t; the snapshot just before it is the
        # newest strictly-before-t state.
        idx = bisect_left(hist.times, t)
        if idx == 0:
            return None
        return dict(hist.rows[idx - 1])
