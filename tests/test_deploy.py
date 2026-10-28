"""Smoke tests against a running deployment.

Skipped unless DEICM_URL is set. scripts/deploy.sh runs them after `fly deploy`:

    DEICM_URL=https://de-icm.fly.dev DEICM_AUTH=user:pass uv run pytest -m deploy
"""

import os

import pytest

httpx = pytest.importorskip("httpx")

URL = os.environ.get("DEICM_URL", "").rstrip("/")
AUTH = os.environ.get("DEICM_AUTH", "")

pytestmark = [
    pytest.mark.deploy,
    pytest.mark.skipif(not URL, reason="set DEICM_URL to run deploy smoke tests"),
]


@pytest.fixture(scope="module")
def client():
    kw = {}
    if AUTH:
        user, _, password = AUTH.partition(":")
        kw["auth"] = (user, password)
    with httpx.Client(base_url=URL, timeout=60, **kw) as c:  # first request may cold-start a machine
        yield c


def test_healthz_is_open():
    r = httpx.get(f"{URL}/healthz", timeout=60)
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_pages_need_auth():
    for path in ("/", "/api/payouts?entrants=100"):
        assert httpx.get(f"{URL}{path}", timeout=60).status_code == 401


@pytest.mark.skipif(not AUTH, reason="set DEICM_AUTH=user:pass")
def test_index_shows_bubble_factor_beside_you_need(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "your BF" in r.text and "you need" in r.text and "Typical field" in r.text


@pytest.mark.skipif(not AUTH, reason="set DEICM_AUTH=user:pass")
def test_table_satellite_bubble(client):
    # Classic ICM: 10 left, 9 equal seats, two 10-chip stacks all-in: caller needs 91.6%.
    body = {"stacks": [10] * 8 + [5, 15], "prizes": [1] * 9, "hero": 0}
    r = client.post("/api/table", json=body)
    assert r.status_code == 200
    opp = next(o for o in r.json()["opponents"] if o["seat"] == 1)
    assert opp["you_need"] == pytest.approx(0.916, abs=1e-3)
    assert opp["your_bf"] == pytest.approx(10.85, abs=0.01)


@pytest.mark.skipif(not AUTH, reason="set DEICM_AUTH=user:pass")
def test_payouts(client):
    r = client.get("/api/payouts", params={"entrants": 582, "paid_pct": 12.2})
    assert r.status_code == 200
    d = r.json()
    assert d["places_paid"] == 71 and sum(d["percent"]) == pytest.approx(100)
    assert d["percent"][0] / d["percent"][8] == pytest.approx(10, rel=1e-3)


@pytest.mark.skipif(not AUTH, reason="set DEICM_AUTH=user:pass")
def test_photo_import_is_configured(client):
    # A non-image upload is rejected with 422 when the API key is set, 503 when it is not.
    r = client.post("/api/extract", files={"image": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422
