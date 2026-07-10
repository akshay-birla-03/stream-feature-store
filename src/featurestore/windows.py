"""Time-window definitions for streaming feature aggregation.

A window answers a single question: *given the current watermark time
``now`` and an event at time ``t``, is that event still inside the window?*

Two flavours are supported:

``SlidingWindow``
    A continuously moving window of fixed ``duration``. An event at time
    ``t`` is a member iff ``now - duration < t <= now``. Sliding windows are
    the natural choice for "features over the last N seconds/minutes/hours".

``TumblingWindow``
    A non-overlapping, gap-free series of fixed-size buckets. Every instant
    belongs to exactly one bucket. Tumbling windows are useful for periodic
    resets (e.g. "count per calendar hour").

Durations are expressed in **seconds** as plain floats so the module stays
dependency-free and easy to reason about. Helpers are provided to convert
``pandas.Timestamp`` / ``datetime`` values to epoch seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

__all__ = [
    "SlidingWindow",
    "TumblingWindow",
    "to_epoch_seconds",
]


def to_epoch_seconds(value) -> float:
    """Coerce a timestamp-like value to epoch seconds (float).

    Accepts ``int``/``float`` (already epoch seconds), ``datetime`` and
    ``pandas.Timestamp``. Naive datetimes are treated as UTC.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts.timestamp()
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise TypeError(f"Cannot convert {type(value)!r} to epoch seconds")


@dataclass(frozen=True)
class SlidingWindow:
    """A fixed-size window that continuously slides with the clock.

    Membership rule (half-open, trailing): an event at ``t`` is in the window
    evaluated at ``now`` iff ``now - duration < t <= now``.

    The leading edge is inclusive (an event exactly at ``now`` counts) and the
    trailing edge is exclusive (an event exactly ``duration`` old has just
    expired). This convention makes eviction deterministic and avoids
    double-counting an event on the tick it ages out.
    """

    duration: float
    name: str = "sliding"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    def contains(self, event_time, now) -> bool:
        t = to_epoch_seconds(event_time)
        n = to_epoch_seconds(now)
        return (n - self.duration) < t <= n

    def start(self, now) -> float:
        """Exclusive lower bound of the window at ``now`` (epoch seconds)."""
        return to_epoch_seconds(now) - self.duration


@dataclass(frozen=True)
class TumblingWindow:
    """A non-overlapping bucketed window of fixed ``duration``.

    Buckets are aligned to ``origin`` (default epoch 0). Bucket index of a
    time ``t`` is ``floor((t - origin) / duration)``. Two events share a
    window iff they map to the same bucket index.
    """

    duration: float
    origin: float = 0.0
    name: str = "tumbling"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    def bucket_index(self, event_time) -> int:
        t = to_epoch_seconds(event_time)
        return int((t - self.origin) // self.duration)

    def bucket_bounds(self, event_time) -> tuple[float, float]:
        """Return ``[start, end)`` epoch-second bounds of the event's bucket."""
        idx = self.bucket_index(event_time)
        start = self.origin + idx * self.duration
        return start, start + self.duration

    def contains(self, event_time, now) -> bool:
        """True iff ``event_time`` falls in the same bucket as ``now``."""
        return self.bucket_index(event_time) == self.bucket_index(now)
