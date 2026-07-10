from datetime import datetime, timezone

import pandas as pd
import pytest

from featurestore.windows import SlidingWindow, TumblingWindow, to_epoch_seconds


def test_to_epoch_seconds_numeric():
    assert to_epoch_seconds(10) == 10.0
    assert to_epoch_seconds(3.5) == 3.5


def test_to_epoch_seconds_datetime_and_timestamp():
    dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    assert to_epoch_seconds(dt) == to_epoch_seconds(ts)
    # Naive treated as UTC.
    assert to_epoch_seconds(datetime(2020, 1, 1)) == to_epoch_seconds(dt)


def test_to_epoch_seconds_rejects_garbage():
    with pytest.raises(TypeError):
        to_epoch_seconds("not-a-time")


def test_sliding_membership_half_open():
    w = SlidingWindow(duration=10.0)
    now = 100.0
    # trailing edge exclusive: t == now-duration is OUT
    assert not w.contains(90.0, now)
    assert w.contains(90.0001, now)
    # leading edge inclusive: t == now is IN
    assert w.contains(100.0, now)
    # future is out
    assert not w.contains(100.5, now)


def test_sliding_start_and_validation():
    w = SlidingWindow(duration=5.0)
    assert w.start(20.0) == 15.0
    with pytest.raises(ValueError):
        SlidingWindow(duration=0)


def test_tumbling_bucket_index_and_bounds():
    w = TumblingWindow(duration=60.0)
    assert w.bucket_index(0) == 0
    assert w.bucket_index(59.9) == 0
    assert w.bucket_index(60.0) == 1
    assert w.bucket_bounds(65.0) == (60.0, 120.0)


def test_tumbling_contains_same_bucket():
    w = TumblingWindow(duration=60.0)
    assert w.contains(10.0, 50.0)  # both bucket 0
    assert not w.contains(10.0, 70.0)  # bucket 0 vs bucket 1
    with pytest.raises(ValueError):
        TumblingWindow(duration=-1)
