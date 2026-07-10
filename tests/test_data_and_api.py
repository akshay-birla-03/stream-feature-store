import pandas as pd
from fastapi.testclient import TestClient

from featurestore import default_feature_specs, generate_events, generate_labels
from featurestore.api import create_app
from featurestore.store import FeatureStore


def test_generate_events_is_sorted_and_deterministic():
    a = generate_events(n_users=3, n_events=50, seed=1)
    b = generate_events(n_users=3, n_events=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    # non-decreasing timestamps
    assert a["event_timestamp"].is_monotonic_increasing
    assert (a["amount"] > 0).all()
    assert set(a.columns) == {"entity_id", "event_timestamp", "amount", "merchant"}


def test_generate_labels_within_span():
    events = generate_events(n_users=3, n_events=50, seed=2)
    labels = generate_labels(events, n_labels=10, seed=5)
    assert len(labels) == 10
    assert labels["event_timestamp"].is_monotonic_increasing
    assert set(labels["entity_id"]).issubset(set(events["entity_id"]))


def _client():
    events = generate_events(n_users=4, n_events=120, seed=11)
    store = FeatureStore(default_feature_specs()).ingest(events)
    return TestClient(create_app(store)), store


def test_health_endpoint():
    client, store = _client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_entities"] == len(store.engine.entities())
    assert body["features"] == store.feature_names


def test_features_endpoint_known_entity():
    client, store = _client()
    entity = store.engine.entities()[0]
    r = client.get(f"/features/{entity}")
    assert r.status_code == 200
    body = r.json()
    assert body["entity_id"] == entity
    assert set(body["features"].keys()) == set(store.feature_names)


def test_features_endpoint_unknown_entity_404():
    client, _ = _client()
    r = client.get("/features/does_not_exist")
    assert r.status_code == 404


def test_online_features_match_store():
    client, store = _client()
    entity = store.engine.entities()[0]
    api_feats = client.get(f"/features/{entity}").json()["features"]
    direct = store.get_online_features(entity)
    for k, v in direct.items():
        assert api_feats[k] == v
