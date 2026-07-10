"""stream-feature-store: streaming feature engineering + point-in-time store.

Public API
----------
* Windows:      :class:`SlidingWindow`, :class:`TumblingWindow`
* Aggregators:  :class:`Count`, :class:`Sum`, :class:`Mean`, :class:`Max`,
                :class:`Min`, :class:`DistinctCount`, :func:`make_aggregator`
* Engine:       :class:`FeatureSpec`, :class:`StreamEngine`
* Store:        :class:`FeatureStore`
* Data:         :func:`generate_events`, :func:`generate_labels`
* Helpers:      :func:`default_feature_specs`
"""

from __future__ import annotations

from .aggregations import (
    Count,
    DistinctCount,
    Max,
    Mean,
    Min,
    Sum,
    make_aggregator,
)
from .data import generate_events, generate_labels
from .engine import FeatureSpec, StreamEngine
from .store import FeatureStore
from .windows import SlidingWindow, TumblingWindow

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SlidingWindow",
    "TumblingWindow",
    "Count",
    "Sum",
    "Mean",
    "Max",
    "Min",
    "DistinctCount",
    "make_aggregator",
    "FeatureSpec",
    "StreamEngine",
    "FeatureStore",
    "generate_events",
    "generate_labels",
    "default_feature_specs",
]


def default_feature_specs() -> list[FeatureSpec]:
    """A representative feature set over transaction events.

    Mixes sliding and tumbling windows and several aggregations so demos and
    the API exercise the full surface.
    """
    win_5m = SlidingWindow(duration=300.0, name="5m")
    win_1h = SlidingWindow(duration=3600.0, name="1h")
    tumble_1h = TumblingWindow(duration=3600.0, name="hourly")
    return [
        FeatureSpec("txn_count_5m", "count", win_5m),
        FeatureSpec("amount_sum_5m", "sum", win_5m, field="amount"),
        FeatureSpec("amount_mean_1h", "mean", win_1h, field="amount"),
        FeatureSpec("amount_max_1h", "max", win_1h, field="amount"),
        FeatureSpec("txn_count_hour", "count", tumble_1h),
        FeatureSpec("distinct_merchants_1h", "distinct_count", win_1h, field="merchant"),
    ]
