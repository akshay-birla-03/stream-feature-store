"""Command-line demo for stream-feature-store.

Generates a synthetic transaction stream, ingests it, then prints:

1. Online features for a sample of entities (what serving would return now).
2. A small point-in-time training set built from label rows -- demonstrating
   the leakage-safe historical join.

Run via the installed console script::

    featurestore --n-users 5 --n-events 300 --n-labels 8
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import default_feature_specs, generate_events, generate_labels
from .store import FeatureStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="featurestore", description=__doc__)
    p.add_argument("--n-users", type=int, default=5)
    p.add_argument("--n-events", type=int, default=300)
    p.add_argument("--n-labels", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    events = generate_events(
        n_users=args.n_users, n_events=args.n_events, seed=args.seed
    )
    store = FeatureStore(default_feature_specs()).ingest(events)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    print("=" * 72)
    print(f"Ingested {store.engine.n_events} events "
          f"for {len(store.engine.entities())} entities.")
    print("Features:", ", ".join(store.feature_names))

    print("\n--- ONLINE FEATURES (latest known values) ---")
    for entity in store.engine.entities()[: min(3, args.n_users)]:
        feats = store.get_online_features(entity)
        pretty = {k: (round(v, 2) if isinstance(v, float) else v)
                  for k, v in feats.items()}
        print(f"  {entity}: {pretty}")

    print("\n--- OFFLINE POINT-IN-TIME TRAINING SET (leakage-safe) ---")
    labels = generate_labels(events, n_labels=args.n_labels, seed=args.seed + 1)
    training = store.get_historical_features(labels)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(training.to_string(index=False))

    print("\nEach feature above is the value known STRICTLY BEFORE the row's")
    print("event_timestamp -- no future events leak into the training set.")
    print("=" * 72)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
