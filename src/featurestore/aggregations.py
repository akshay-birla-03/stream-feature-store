"""Incremental aggregators for windowed streaming features.

Each aggregator maintains internal state so that adding or evicting a single
event is cheap (amortised O(1) for most, O(log n) for order-sensitive ones)
rather than recomputing over the whole window every tick.

All aggregators share a tiny protocol:

* ``add(value)``     -- incorporate a new event value into the state.
* ``evict(value)``   -- remove a previously added event value (it aged out of
                        the window). Callers must only evict values that were
                        previously added; the aggregators trust this contract.
* ``value``          -- the current aggregate as a plain Python scalar (or
                        ``None`` when the window is empty, where meaningful).
* ``count``          -- number of events currently retained.

The ``Max``/``Min`` aggregators keep a sorted multiset so eviction of an
arbitrary (not necessarily extreme) value stays correct and efficient. This
matters for sliding windows where the oldest event -- which may or may not be
the current max -- expires each tick.
"""

from __future__ import annotations

from bisect import insort
from collections import Counter

__all__ = [
    "Aggregator",
    "Count",
    "Sum",
    "Mean",
    "Max",
    "Min",
    "DistinctCount",
    "AGGREGATOR_REGISTRY",
    "make_aggregator",
]


class Aggregator:
    """Base class defining the incremental aggregator protocol."""

    name = "base"

    def add(self, value) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def evict(self, value) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def value(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def copy(self) -> Aggregator:  # pragma: no cover - abstract
        raise NotImplementedError


class Count(Aggregator):
    """Number of events currently in the window."""

    name = "count"

    def __init__(self) -> None:
        self._n = 0

    def add(self, value) -> None:
        self._n += 1

    def evict(self, value) -> None:
        self._n -= 1
        if self._n < 0:
            raise ValueError("evicted more events than were added")

    @property
    def count(self) -> int:
        return self._n

    @property
    def value(self) -> int:
        return self._n

    def copy(self) -> Count:
        c = Count()
        c._n = self._n
        return c


class Sum(Aggregator):
    """Running sum of event values."""

    name = "sum"

    def __init__(self) -> None:
        self._total = 0.0
        self._n = 0

    def add(self, value) -> None:
        self._total += float(value)
        self._n += 1

    def evict(self, value) -> None:
        self._total -= float(value)
        self._n -= 1

    @property
    def count(self) -> int:
        return self._n

    @property
    def value(self) -> float:
        return self._total

    def copy(self) -> Sum:
        s = Sum()
        s._total = self._total
        s._n = self._n
        return s


class Mean(Aggregator):
    """Running arithmetic mean; ``None`` on an empty window."""

    name = "mean"

    def __init__(self) -> None:
        self._total = 0.0
        self._n = 0

    def add(self, value) -> None:
        self._total += float(value)
        self._n += 1

    def evict(self, value) -> None:
        self._total -= float(value)
        self._n -= 1
        if self._n < 0:
            raise ValueError("evicted more events than were added")

    @property
    def count(self) -> int:
        return self._n

    @property
    def value(self) -> float | None:
        if self._n == 0:
            return None
        return self._total / self._n

    def copy(self) -> Mean:
        m = Mean()
        m._total = self._total
        m._n = self._n
        return m


class _SortedMultiset(Aggregator):
    """Shared machinery for Max/Min via a sorted list of values."""

    def __init__(self) -> None:
        self._items: list = []

    def add(self, value) -> None:
        insort(self._items, float(value))

    def evict(self, value) -> None:
        v = float(value)
        # Locate via bisect for O(log n) find, then remove (O(n) shift). Kept
        # simple and correct; windows are typically small.
        from bisect import bisect_left

        i = bisect_left(self._items, v)
        if i >= len(self._items) or self._items[i] != v:
            raise ValueError(f"value {v!r} not present to evict")
        self._items.pop(i)

    @property
    def count(self) -> int:
        return len(self._items)

    def copy(self):
        obj = type(self)()
        obj._items = list(self._items)
        return obj


class Max(_SortedMultiset):
    """Maximum event value; ``None`` on an empty window."""

    name = "max"

    @property
    def value(self) -> float | None:
        return self._items[-1] if self._items else None


class Min(_SortedMultiset):
    """Minimum event value; ``None`` on an empty window."""

    name = "min"

    @property
    def value(self) -> float | None:
        return self._items[0] if self._items else None


class DistinctCount(Aggregator):
    """Number of distinct values currently in the window.

    Backed by a ``Counter`` multiset so that evicting one occurrence of a
    repeated value does not prematurely drop it from the distinct set.
    """

    name = "distinct_count"

    def __init__(self) -> None:
        self._counts: Counter = Counter()

    def add(self, value) -> None:
        self._counts[value] += 1

    def evict(self, value) -> None:
        if self._counts[value] <= 0:
            raise ValueError(f"value {value!r} not present to evict")
        self._counts[value] -= 1
        if self._counts[value] == 0:
            del self._counts[value]

    @property
    def count(self) -> int:
        return sum(self._counts.values())

    @property
    def value(self) -> int:
        return len(self._counts)

    def copy(self) -> DistinctCount:
        d = DistinctCount()
        d._counts = self._counts.copy()
        return d


AGGREGATOR_REGISTRY = {
    "count": Count,
    "sum": Sum,
    "mean": Mean,
    "max": Max,
    "min": Min,
    "distinct_count": DistinctCount,
}


def make_aggregator(name: str) -> Aggregator:
    """Instantiate an aggregator by its registry name."""
    try:
        return AGGREGATOR_REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(
            f"unknown aggregator {name!r}; choices: {sorted(AGGREGATOR_REGISTRY)}"
        ) from exc
