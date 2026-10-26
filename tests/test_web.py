import base64

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from deicm import icm  # noqa: E402
from deicm.web.app import create_app  # noqa: E402

BODY = {"stacks": [1000, 2500, 400, 3000], "prizes": [50, 30, 20], "hero": 0}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("AUTH_USERS", raising=False)
    return TestClient(create_app())


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "DE-ICM" in r.text


def test_table_keys_and_equity(client):
    r = client.post("/api/table", json=BODY)
    assert r.status_code == 200
    d = r.json()
    assert {"seats", "opponents", "required_equity", "grid", "pool", "hero", "ms"} <= d.keys()
    assert d["pool"] == 100
    ref = icm(BODY["stacks"], BODY["prizes"])
    assert len(d["seats"]) == 4
    for i, s in enumerate(d["seats"]):
        assert {"stack", "chip_ev", "equity", "delta", "itm", "p_first"} <= s.keys()
        assert s["equity"] == pytest.approx(ref.equities[i], abs=1e-12)
        assert s["chip_ev"] == pytest.approx(ref.chip_ev[i])
        assert s["p_first"] == pytest.approx(ref.placements[i, 0], abs=1e-12)
    assert [o["seat"] for o in d["opponents"]] == [1, 2, 3]
    for o in d["opponents"]:
        assert {"you_need", "they_need", "your_bf", "their_bf", "you_gain", "you_lose"} <= o.keys()
        assert 0 < o["you_need"] < 1
        assert o["you_need"] == pytest.approx(d["required_equity"][0][o["seat"]])
        assert o["you_need"] == pytest.approx(o["you_lose"] / (o["you_gain"] + o["you_lose"]))
    assert d["required_equity"][0][0] is None


@pytest.mark.parametrize(
    "patch",
    [
        {"stacks": [1000]},
        {"stacks": [1000, -5]},
        {"stacks": [1000, 2000], "prizes": [5, 3, 2]},  # more prizes than players
        {"prizes": [20, 30]},  # increasing
        {"hero": 7},
        {"pot": -1},
        {"others": {"count": 0, "average": 100}},
        {"others": {"count": 10, "average": -3}},
    ],
)
def test_validation_errors_are_422(client, patch):
    r = client.post("/api/table", json={**BODY, **patch})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_missing_body_field_is_422(client):
    r = client.post("/api/table", json={"stacks": [1, 2]})
    assert r.status_code == 422 and "prizes" in r.json()["detail"]


def test_basic_auth(monkeypatch):
    monkeypatch.setenv("AUTH_USERS", "jesse:pw1,guest:pw2")
    c = TestClient(create_app())
    assert c.get("/healthz").status_code == 200
    assert c.get("/").status_code == 401
    assert c.post("/api/table", json=BODY).status_code == 401

    def hdr(u, p):
        return {"Authorization": "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()}

    assert c.get("/", headers=hdr("jesse", "wrong")).status_code == 401
    assert c.get("/", headers=hdr("nobody", "pw1")).status_code == 401
    assert c.get("/", headers=hdr("jesse", "pw2")).status_code == 401
    assert c.get("/", headers={"Authorization": "Basic !!!"}).status_code == 401
    assert c.get("/", headers=hdr("jesse", "pw1")).status_code == 200
    assert c.get("/", headers=hdr("guest", "pw2")).status_code == 200
    assert c.post("/api/table", json=BODY, headers=hdr("guest", "pw2")).status_code == 200


def test_open_when_auth_unset(client):
    assert client.get("/").status_code == 200
    assert client.post("/api/table", json=BODY).status_code == 200


def test_others_appends_players(client):
    body = {**BODY, "prizes": [50, 30, 20, 10], "others": {"count": 6, "average": 1500}}
    d = client.post("/api/table", json=body).json()
    assert d["n"] == 10 and len(d["seats"]) == 4  # only listed seats are shown
    ref = icm([1000, 2500, 400, 3000] + [1500] * 6, [50, 30, 20, 10])
    for i, s in enumerate(d["seats"]):
        assert s["equity"] == pytest.approx(ref.equities[i], abs=1e-12)
    # prizes longer than the listed field are only legal because of `others`
    assert client.post("/api/table", json={**body, "others": None}).status_code == 200


def test_pot_lowers_you_need(client):
    base = client.post("/api/table", json=BODY).json()["opponents"]
    dead = client.post("/api/table", json={**BODY, "pot": 600}).json()["opponents"]
    for a, b in zip(base, dead):
        assert b["you_need"] < a["you_need"]
