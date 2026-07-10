"""Feature store: online serving + leakage-safe offline point-in-time joins.

``FeatureStore`` wraps a :class:`~featurestore.engine.StreamEngine` with the
two access patterns every production feature store must offer:

Online serving
    ``get_online_features(entity_id)`` returns the latest materialised feature
    vector for an entity -- the value an inference service would read at
    request time. O(1) dictionary lookup.

Offline training-set generation
    ``get_historical_features(label_df)`` performs a **point-in-time join**.
    For each ``(entity_id, event_timestamp)`` label row it attaches the
    feature values that were known *strictly before* that timestamp. This is
    the headline capability: it guarantees the training set contains exactly
    what online serving would have returned at each label's moment -- no
    future-data leakage, no online/offline skew.

Why point-in-time correctness matters
--------------------------------------
A naive "join features to labels on entity_id" attaches whatever the feature
value is *now*, which for a historical label silently includes information
from after the label was created. A model trained on such data looks great
offline and collapses in production, because at serving time the future isn't
available. The point-in-time join reproduces the exact information frontier of
each label.
"""

from __future__ import annotations

import pandas as pd

from .engine import FeatureSpec, StreamEngine

__all__ = ["FeatureStore"]


class FeatureStore:
    """High-level facade over a streaming feature engine."""

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self.engine = StreamEngine(specs)
        self.feature_names = self.engine.feature_names

    # -- construction ------------------------------------------------------
    def ingest(
        self,
        events: pd.DataFrame,
        entity_col: str = "entity_id",
        time_col: str = "event_timestamp",
    ) -> FeatureStore:
        """Ingest an event DataFrame; returns ``self`` for chaining."""
        self.engine.ingest_frame(events, entity_col=entity_col, time_col=time_col)
        return self

    # -- online path -------------------------------------------------------
    def get_online_features(self, entity_id) -> dict:
        """Latest feature values for ``entity_id``.

        Unknown entities yield an all-``None`` vector so callers get a stable
        schema rather than a ``KeyError``.
        """
        state = self.engine.get_online_state(entity_id)
        if state is None:
            return {name: None for name in self.feature_names}
        return state

    # -- offline path (point-in-time join) --------------------------------
    def get_historical_features(
        self,
        label_df: pd.DataFrame,
        entity_col: str = "entity_id",
        time_col: str = "event_timestamp",
    ) -> pd.DataFrame:
        """Leakage-safe point-in-time join of features onto label rows.

        Parameters
        ----------
        label_df:
            Must contain ``entity_col`` and ``time_col``. Any additional
            columns (e.g. a ``label``) are carried through untouched.

        Returns
        -------
        DataFrame
            ``label_df`` (row order and index preserved) with one column added
            per feature, populated with the value known strictly before each
            row's ``event_timestamp``. Missing values (no prior events) are
            ``NaN``/``None``.
        """
        for col in (entity_col, time_col):
            if col not in label_df.columns:
                raise KeyError(f"label_df is missing required column {col!r}")

        feature_rows = []
        for entity, ts in zip(label_df[entity_col], label_df[time_col]):
            asof = self.engine.features_asof(entity, ts)
            if asof is None:
                asof = {name: None for name in self.feature_names}
            feature_rows.append(asof)

        features = pd.DataFrame(feature_rows, columns=self.feature_names)
        features.index = label_df.index
        # Concatenate keeping original label columns first.
        return pd.concat([label_df, features], axis=1)
