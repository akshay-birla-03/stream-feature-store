import random

import pytest

from featurestore.aggregations import (
    Count,
    DistinctCount,
    Max,
    Mean,
    Min,
    Sum,
    make_aggregator,
)


def brute(name, values):
    if not values:
        if name in ("mean", "max", "min"):
            return None
        if name in ("count", "distinct_count"):
            return 0
        return 0.0
    if name == "count":
        return len(values)
    if name == "sum":
        return sum(values)
    if name == "mean":
        return sum(values) / len(values)
    if name == "max":
        return max(values)
    if name == "min":
        return min(values)
    if name == "distinct_count":
        return len(set(values))
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["count", "sum", "mean", "max", "min", "distinct_count"])
def test_add_matches_bruteforce(name):
    agg = make_aggregator(name)
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    live = []
    for v in values:
        agg.add(v)
        live.append(v)
        assert agg.value == brute(name, live)


@pytest.mark.parametrize("name", ["count", "sum", "mean", "max", "min", "distinct_count"])
def test_sliding_add_evict_matches_bruteforce(name):
    """Simulate a size-3 sliding window: add newest, evict oldest, compare."""
    agg = make_aggregator(name)
    rng = random.Random(42)
    stream = [rng.randint(0, 5) for _ in range(60)]
    window = []
    for i, v in enumerate(stream):
        agg.add(v)
        window.append(v)
        if len(window) > 3:
            old = window.pop(0)
            agg.evict(old)
        assert agg.value == brute(name, window), f"{name} failed at step {i}"


def test_empty_semantics():
    assert Mean().value is None
    assert Max().value is None
    assert Min().value is None
    assert Count().value == 0
    assert Sum().value == 0.0
    assert DistinctCount().value == 0


def test_distinct_count_multiset_eviction():
    d = DistinctCount()
    d.add("a")
    d.add("a")
    d.add("b")
    assert d.value == 2
    d.evict("a")  # still one 'a' left
    assert d.value == 2
    d.evict("a")
    assert d.value == 1


def test_max_min_evict_non_extreme():
    mx = Max()
    for v in [5, 1, 9, 3]:
        mx.add(v)
    assert mx.value == 9
    mx.evict(1)  # evicting a non-max must not change max
    assert mx.value == 9
    mx.evict(9)
    assert mx.value == 5


def test_evict_errors_and_over_evict():
    with pytest.raises(ValueError):
        Count().evict(1)
    with pytest.raises(ValueError):
        Max().evict(7)
    with pytest.raises(ValueError):
        DistinctCount().evict("x")


def test_copy_is_independent():
    s = Sum()
    s.add(10)
    s2 = s.copy()
    s2.add(5)
    assert s.value == 10
    assert s2.value == 15


def test_make_aggregator_unknown():
    with pytest.raises(KeyError):
        make_aggregator("median")
