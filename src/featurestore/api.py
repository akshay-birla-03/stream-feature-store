"""FastAPI online-serving app.

Exposes a minimal read path over a :class:`FeatureStore` materialised at
startup from a synthetic stream. In production the store would be backed by a
real online DB; here it demonstrates the serving contract and response schema.

Endpoints
---------
``GET /health``               -> liveness + entity/feature counts.
``GET /features/{entity_id}`` -> latest online feature vector for an entity.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import default_feature_specs, generate_events
from .store import FeatureStore


def build_store() -> FeatureStore:
    """Materialise a demo store from a deterministic synthetic stream."""
    events = generate_events(n_users=5, n_events=300, seed=7)
    return FeatureStore(default_feature_specs()).ingest(events)


def create_app(store: FeatureStore | None = None) -> FastAPI:
    """Application factory (keeps the store injectable for tests)."""
    store = store or build_store()
    app = FastAPI(
        title="stream-feature-store",
        description="Online serving for streaming windowed features.",
        version="0.1.0",
    )
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        engine = store.engine
        return {
            "status": "ok",
            "n_entities": len(engine.entities()),
            "n_events": engine.n_events,
            "features": store.feature_names,
        }

    @app.get("/features/{entity_id}")
    def get_features(entity_id: str) -> dict:
        engine = store.engine
        if engine.get_online_state(entity_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown entity {entity_id!r}")
        return {
            "entity_id": entity_id,
            "features": store.get_online_features(entity_id),
        }

    return app


app = create_app()
