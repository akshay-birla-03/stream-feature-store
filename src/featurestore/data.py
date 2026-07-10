"""Synthetic event-stream and label generators (fully offline).

The generators fabricate a plausible stream of user *transaction* events --
each with an entity (user), a timestamp, an amount and a merchant category --
plus a set of label rows at arbitrary query times. Everything is driven by a
seedable ``numpy`` RNG so tests and demos are deterministic.

The event timestamps are returned as epoch seconds (float) for simplicity and
so the point-in-time semantics are easy to reason about; a ``base_time`` lets
callers anchor them to a wall-clock instant if desired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["generate_events", "generate_labels", "MERCHANTS"]

MERCHANTS = ["grocery", "fuel", "electronics", "dining", "travel"]


def generate_events(
    n_users: int = 5,
    n_events: int = 200,
    seed: int = 0,
    base_time: float = 0.0,
    max_gap: float = 60.0,
) -> pd.DataFrame:
    """Generate a time-sorted transaction event stream.

    Columns: ``entity_id`` (``user_{i}``), ``event_timestamp`` (epoch
    seconds), ``amount`` (positive float), ``merchant`` (categorical).

    Timestamps are globally non-decreasing (required by the engine): we draw
    per-event gaps and take a cumulative sum.
    """
    rng = np.random.default_rng(seed)
    users = [f"user_{i}" for i in range(n_users)]

    gaps = rng.uniform(0.0, max_gap, size=n_events)
    times = base_time + np.cumsum(gaps)
    entity = rng.choice(users, size=n_events)
    # Per-user amount scale so aggregates differ across entities.
    scale = {u: rng.uniform(10.0, 100.0) for u in users}
    amounts = np.array(
        [round(abs(rng.normal(scale[u], scale[u] * 0.3)) + 1.0, 2) for u in entity]
    )
    merchants = rng.choice(MERCHANTS, size=n_events)

    return pd.DataFrame(
        {
            "entity_id": entity,
            "event_timestamp": times,
            "amount": amounts,
            "merchant": merchants,
        }
    )


def generate_labels(
    events: pd.DataFrame,
    n_labels: int = 20,
    seed: int = 1,
    horizon: float = 30.0,
) -> pd.DataFrame:
    """Generate label rows at query times drawn from the event span.

    Each label picks a random entity present in ``events`` and a query
    timestamp uniformly within the event time span (nudged by ``horizon`` so
    some queries fall between events). A synthetic binary ``label`` is
    attached. The returned frame is sorted by time.
    """
    rng = np.random.default_rng(seed)
    entities = events["entity_id"].unique()
    t_min = float(events["event_timestamp"].min())
    t_max = float(events["event_timestamp"].max())

    ent = rng.choice(entities, size=n_labels)
    times = rng.uniform(t_min, t_max + horizon, size=n_labels)
    labels = rng.integers(0, 2, size=n_labels)

    df = pd.DataFrame(
        {
            "entity_id": ent,
            "event_timestamp": times,
            "label": labels,
        }
    )
    return df.sort_values("event_timestamp").reset_index(drop=True)
