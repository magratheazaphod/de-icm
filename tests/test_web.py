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


# ---------- photo import (no live API calls) ----------
from deicm.web.extract import Payout, TournamentState, derive_prefill  # noqa: E402


def _state(**kw):
    return TournamentState(source_kind="pokeratlas", confidence="high", **kw)


def test_prefill_per_player_ante_and_ranges():
    s = _state(
        small_blind=200, big_blind=400, ante=50, ante_kind="per_player",
        players_remaining=60, average_stack=20000,
        payouts=[Payout(place_from=4, place_to=6, amount=300), Payout(place_from=1, place_to=1, amount=1000),
                 Payout(place_from=2, place_to=2, amount=600), Payout(place_from=3, place_to=3, amount=450)],
    )
    pf = derive_prefill(s)
    assert pf["prizes"] == [1000, 600, 450, 300, 300, 300]
    assert pf["pot"] == 200 + 400 + 50 * 9
    assert pf["others"] == {"count": 51, "average": 20000}
    assert pf["players_remaining"] == 60 and pf["average_stack"] == 20000


def test_prefill_big_blind_ante_and_none():
    bba = derive_prefill(_state(small_blind=100, big_blind=200, ante=200, ante_kind="big_blind_ante"))
    assert bba["pot"] == 500
    none = derive_prefill(_state(small_blind=100, big_blind=200, ante=75, ante_kind="none"))
    assert none["pot"] == 300


def test_prefill_missing_fields():
    pf = derive_prefill(_state())
    assert pf == {"prizes": None, "pot": None, "others": None, "players_remaining": None, "average_stack": None}
    assert derive_prefill(_state(players_remaining=40))["others"] is None
    assert derive_prefill(_state(big_blind=400))["pot"] == 400


def test_extract_503_when_key_unset(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/extract", files={"image": ("a.png", b"x", "image/png")})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_extract_rejects_bad_uploads(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    r = client.post("/api/extract", files={"image": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 422
    big = b"0" * (8 * 1024 * 1024 + 1)
    r = client.post("/api/extract", files={"image": ("a.png", big, "image/png")})
    assert r.status_code == 422 and "8 MB" in r.json()["detail"]


def test_extract_requires_auth(monkeypatch):
    monkeypatch.setenv("AUTH_USERS", "a:b")
    c = TestClient(create_app())
    assert c.post("/api/extract", files={"image": ("a.png", b"x", "image/png")}).status_code == 401


def test_extract_error_mapping(client, monkeypatch):
    import anthropic
    import httpx

    import deicm.web.app as appmod
    from deicm.web.extract import RefusalError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    files = {"image": ("a.png", b"x", "image/png")}
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def raiser(exc):
        def f(*a, **k):
            raise exc
        return f

    monkeypatch.setattr(appmod, "extract_state", raiser(RefusalError("nope")))
    r = client.post("/api/extract", files=files)
    assert r.status_code == 422 and "nope" in r.json()["detail"]

    rl = anthropic.RateLimitError("slow", response=httpx.Response(429, request=req), body=None)
    st = anthropic.InternalServerError("boom", response=httpx.Response(500, request=req), body=None)
    cn = anthropic.APIConnectionError(request=req)
    for exc in (rl, st, cn):
        monkeypatch.setattr(appmod, "extract_state", raiser(exc))
        assert client.post("/api/extract", files=files).status_code == 502
