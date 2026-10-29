import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deicm.web.app import create_app  # noqa: E402
from deicm.web.store import Store  # noqa: E402

SPOT = {"seats": ["300000", "450000"], "hero": 0, "pay": "50 30 20", "mode": "pct"}


def new_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTH_USERS", raising=False)
    monkeypatch.setenv("DEICM_DB", str(tmp_path / "deicm.db"))
    return TestClient(create_app())


def test_storage_is_503_without_a_database(monkeypatch):
    monkeypatch.delenv("AUTH_USERS", raising=False)
    monkeypatch.delenv("DEICM_DB", raising=False)
    c = TestClient(create_app())
    assert c.get("/api/scenarios").status_code == 503
    assert c.get("/healthz").status_code == 200


def test_tournament_spots_round_trip(client):
    tid = new_id()
    t = client.put(f"/api/tournaments/{tid}", json={"name": "PokerAtlas Tour Austin", "venue": "TCH Social",
                                                    "start_date": "2026-11-10", "entrants": 600, "paid_pct": 12,
                                                    "client_ts": 1})
    assert t.status_code == 200 and t.json()["venue"] == "TCH Social"
    client.put(f"/api/scenarios/{new_id()}", json={"tournament_id": tid, "title": "Bubble jam vs CL",
                                                   "input": SPOT, "client_ts": 1})
    client.put(f"/api/scenarios/{new_id()}", json={"title": "Loose spot", "input": SPOT, "client_ts": 1})

    filed = client.get("/api/scenarios", params={"tournament_id": tid}).json()
    assert [x["title"] for x in filed] == ["Bubble jam vs CL"] and filed[0]["input"] == SPOT
    assert [x["title"] for x in client.get("/api/scenarios", params={"unfiled": True}).json()] == ["Loose spot"]
    assert client.get("/api/tournaments").json()[0]["n_scenarios"] == 1

    # Deleting a tournament keeps its spots, unfiled.
    assert client.delete(f"/api/tournaments/{tid}").status_code == 204
    assert len(client.get("/api/scenarios", params={"unfiled": True}).json()) == 2


def test_replayed_put_is_idempotent(client):
    sid = new_id()
    body = {"title": "a", "input": SPOT, "client_ts": 5}
    first = client.put(f"/api/scenarios/{sid}", json=body).json()
    again = client.put(f"/api/scenarios/{sid}", json=body).json()
    assert again["created_at"] == first["created_at"]
    assert len(client.get("/api/scenarios").json()) == 1


def test_stale_write_does_not_undo_a_newer_one(client):
    tid = new_id()
    client.put(f"/api/tournaments/{tid}", json={"name": "T", "live": {"left": 40}, "client_ts": 200})
    # An older edit, queued offline earlier in the level, arrives late.
    r = client.put(f"/api/tournaments/{tid}", json={"name": "T", "live": {"left": 55}, "client_ts": 100})
    assert r.status_code == 200 and r.json()["live"] == {"left": 40}
    r = client.put(f"/api/tournaments/{tid}", json={"name": "T", "live": {"left": 31}, "client_ts": 300})
    assert r.json()["live"] == {"left": 31}


def test_scenario_needs_existing_tournament(client):
    r = client.put(f"/api/scenarios/{new_id()}", json={"tournament_id": new_id(), "title": "x",
                                                       "input": SPOT, "client_ts": 1})
    assert r.status_code == 409


def test_players_and_unique_nickname(client):
    pid = new_id()
    p = client.put(f"/api/players/{pid}", json={"nickname": "Big Mike", "description": "red hat",
                                                "fold_to_cbet": 0.7, "client_ts": 1})
    assert p.status_code == 200
    dup = client.put(f"/api/players/{new_id()}", json={"nickname": "big mike", "client_ts": 1})
    assert dup.status_code == 409
    r = client.put(f"/api/players/{pid}", json={"nickname": "Big Mike", "description": "red hat",
                                                "fold_to_cbet": None, "client_ts": 2})
    assert r.json()["fold_to_cbet"] is None
    bad = client.put(f"/api/players/{new_id()}", json={"nickname": "x", "fold_to_cbet": 1.5, "client_ts": 1})
    assert bad.status_code == 422


def test_observations_append_and_follow_the_player(client):
    pid, tid = new_id(), new_id()
    client.put(f"/api/players/{pid}", json={"nickname": "P", "client_ts": 1})
    client.put(f"/api/tournaments/{tid}", json={"name": "T", "client_ts": 1})
    oid = new_id()
    body = {"player_id": pid, "tournament_id": tid, "text": "limp-called AQo UTG, L12", "seen_at": 1000}
    assert client.put(f"/api/observations/{oid}", json=body).status_code == 200
    assert client.put(f"/api/observations/{oid}", json=body).status_code == 200  # replay is a no-op
    client.put(f"/api/observations/{new_id()}", json={**body, "text": "folded to cbet", "seen_at": 2000})
    obs = client.get("/api/observations", params={"player_id": pid}).json()
    assert [o["text"] for o in obs] == ["folded to cbet", "limp-called AQo UTG, L12"]
    assert client.delete(f"/api/players/{pid}").status_code == 204
    assert client.get("/api/observations").json() == []


def test_delete_is_idempotent(client):
    sid = new_id()
    assert client.delete(f"/api/scenarios/{sid}").status_code == 204
    client.put(f"/api/scenarios/{sid}", json={"title": "a", "input": SPOT, "client_ts": 1})
    assert client.delete(f"/api/scenarios/{sid}").status_code == 204
    assert client.delete(f"/api/scenarios/{sid}").status_code == 204


def test_bad_ids_and_missing_clock_are_rejected(client):
    assert client.put("/api/scenarios/x", json={"title": "a", "input": SPOT, "client_ts": 1}).status_code == 422
    assert client.put(f"/api/scenarios/{new_id()}", json={"title": "a", "input": SPOT}).status_code == 422


def test_export_has_everything(client):
    pid = new_id()
    client.put(f"/api/tournaments/{new_id()}", json={"name": "T", "client_ts": 1})
    client.put(f"/api/players/{pid}", json={"nickname": "P", "client_ts": 1})
    client.put(f"/api/scenarios/{new_id()}", json={"title": "s", "input": SPOT, "client_ts": 1})
    client.put(f"/api/observations/{new_id()}", json={"player_id": pid, "text": "x", "seen_at": 1})
    d = client.get("/api/export").json()
    assert all(len(d[t]) == 1 for t in ("tournaments", "players", "scenarios", "observations"))
    assert d["scenarios"][0]["input"] == SPOT


def test_reopening_keeps_data_and_schema(tmp_path):
    path = tmp_path / "x.db"
    Store(path).put("players", new_id(), {"nickname": "P"}, 1)
    assert [p["nickname"] for p in Store(path).list("players")] == ["P"]


def test_storage_needs_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("DEICM_DB", str(tmp_path / "deicm.db"))
    monkeypatch.setenv("AUTH_USERS", "u:p")
    c = TestClient(create_app())
    assert c.get("/api/players").status_code == 401
    assert c.get("/api/players", auth=("u", "p")).status_code == 200
