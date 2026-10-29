"""Mobile-first web app for DE-ICM (docs/SPEC.md section 11).

Run with ``deicm-web`` (uvicorn on $PORT, default 8080). Set ``AUTH_USERS`` to
``user:pass,user:pass`` to require HTTP basic auth on every route except
``/healthz``.
"""

from __future__ import annotations

import base64
import binascii
import math
import os
import secrets
import sqlite3
import time
from pathlib import Path as FilePath
from typing import Annotated

import logging

from fastapi import FastAPI, File, HTTPException, Path, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from deicm.core import icm
from deicm.payouts import payout_table
from deicm.web.extract import RefusalError, derive_prefill, extract_state
from deicm.web.store import Store
from deicm.perturb import allin, bubble_factors

INDEX_HTML = FilePath(__file__).parent / "static" / "index.html"
MAX_PLAYERS = 4000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
log = logging.getLogger("deicm.web")


class Others(BaseModel):
    count: int = Field(ge=1, le=MAX_PLAYERS)
    average: float = Field(gt=0, allow_inf_nan=False)


class TableRequest(BaseModel):
    stacks: list[float] = Field(min_length=2, max_length=12)
    prizes: list[float] = Field(min_length=1, max_length=MAX_PLAYERS)
    hero: int = Field(default=0, ge=0)
    pot: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    others: Others | None = None

    @model_validator(mode="after")
    def _check(self) -> "TableRequest":
        if any(not math.isfinite(x) or x <= 0 for x in self.stacks):
            raise ValueError("every stack must be a positive number")
        if any(not math.isfinite(x) or x <= 0 for x in self.prizes):
            raise ValueError("every prize must be a positive number")
        if self.hero >= len(self.stacks):
            raise ValueError(f"hero {self.hero} is not one of the {len(self.stacks)} listed stacks")
        extra = self.others.count if self.others else 0
        if len(self.stacks) + extra > MAX_PLAYERS:
            raise ValueError(f"at most {MAX_PLAYERS} players in total")
        return self


Rate = Field(default=None, ge=0, le=1, allow_inf_nan=False)
RecordId = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]{8,64}$")]


class Write(BaseModel):
    """Every stored write carries the phone's clock, so stale retries lose (last write wins)."""

    client_ts: float = Field(ge=0, allow_inf_nan=False)


class TournamentIn(Write):
    name: str = Field(min_length=1, max_length=120)
    venue: str = Field(default="", max_length=120)
    start_date: str = Field(default="", max_length=10)
    buy_in: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    entrants: int | None = Field(default=None, ge=2, le=100_000)
    paid_pct: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)
    notes: str = Field(default="", max_length=4000)
    live: dict | None = None  # mid-tournament snapshot: level, players left, average stack, ...


class ScenarioIn(Write):
    tournament_id: str | None = None
    title: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=4000)
    input: dict  # the page state, stored as given


class PlayerIn(Write):
    nickname: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=400)
    archetype: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=8000)
    fold_to_cbet: float | None = Rate
    raise_vs_cbet: float | None = Rate
    re_jam_rate: float | None = Rate
    fold_to_iso: float | None = Rate


class ObservationIn(BaseModel):
    player_id: str
    tournament_id: str | None = None
    text: str = Field(min_length=1, max_length=2000)
    seen_at: float = Field(ge=0, allow_inf_nan=False)


def _num(x: float) -> float | None:
    """JSON has no NaN or infinity."""
    x = float(x)
    return x if math.isfinite(x) else None


def compute_table(req: TableRequest) -> dict:
    t0 = time.perf_counter()
    k = len(req.stacks)
    stacks = list(req.stacks)
    if req.others:
        stacks += [req.others.average] * req.others.count
    prizes = req.prizes
    res = icm(stacks, prizes)
    bf = bubble_factors(stacks, prizes, seats=list(range(k)), pot=req.pot)

    hero = req.hero
    seats = []
    for i in range(k):
        eq, ev = float(res.equities[i]), float(res.chip_ev[i])
        seats.append(
            {
                "seat": i,
                "stack": stacks[i],
                "chip_ev": ev,
                "equity": eq,
                "delta": eq - ev,
                "delta_pct": _num((eq - ev) / ev) if ev else None,
                "itm": float(res.itm[i]),
                "p_first": float(res.placements[i, 0]),
            }
        )

    opponents = []
    for j in range(k):
        if j == hero:
            continue
        gain = float(allin(stacks, prizes, hero, j, pot=req.pot).delta[hero])
        lose = -float(allin(stacks, prizes, j, hero, pot=req.pot).delta[hero])
        opponents.append(
            {
                "seat": j,
                "stack": stacks[j],
                "you_need": _num(bf.required_equity[hero, j]),
                "they_need": _num(bf.required_equity[j, hero]),
                "your_bf": _num(bf.bf[hero, j]),
                "their_bf": _num(bf.bf[j, hero]),
                "you_gain": _num(gain),
                "you_lose": _num(lose),
            }
        )

    return {
        "hero": hero,
        "n": len(stacks),
        "pool": float(sum(prizes)),
        "pot": req.pot,
        "grid": {"h": res.grid[0], "M": res.grid[1]},
        "seats": seats,
        "opponents": opponents,
        "required_equity": [[_num(v) for v in row] for row in bf.required_equity],
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def _parse_users(raw: str | None) -> dict[str, str]:
    users: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if part and ":" in part:
            user, _, password = part.partition(":")
            users[user] = password
    return users


def _authorized(header: str | None, users: dict[str, str]) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    user, _, password = decoded.partition(":")
    # Compare against every configured user so timing does not reveal which exist.
    ok = False
    for name, expected in users.items():
        name_ok = secrets.compare_digest(user.encode(), name.encode())
        pass_ok = secrets.compare_digest(password.encode(), expected.encode())
        ok = (name_ok and pass_ok) or ok
    return ok


def create_app() -> FastAPI:
    app = FastAPI(title="DE-ICM", docs_url=None, redoc_url=None, openapi_url=None)
    users = _parse_users(os.environ.get("AUTH_USERS"))
    db_path = os.environ.get("DEICM_DB")
    store = Store(db_path) if db_path else None

    def db() -> Store:
        if store is None:
            raise HTTPException(503, "Saving is not configured on this server (DEICM_DB is not set).")
        return store

    if users:

        @app.middleware("http")
        async def basic_auth(request: Request, call_next):
            if request.url.path != "/healthz" and not _authorized(
                request.headers.get("authorization"), users
            ):
                return Response(
                    "Authentication required",
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="deicm"'},
                )
            return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"] if p != "body")
            msg = str(err["msg"]).removeprefix("Value error, ")
            msgs.append(f"{loc}: {msg}" if loc else msg)
        return JSONResponse({"detail": "; ".join(msgs)}, status_code=422)

    @app.exception_handler(sqlite3.IntegrityError)
    async def _conflict(request: Request, exc: sqlite3.IntegrityError):
        msg = str(exc)
        if "UNIQUE" in msg and "players.nickname" in msg:
            msg = "A player with that nickname already exists."
        elif "FOREIGN KEY" in msg:
            msg = "It refers to a tournament or player that does not exist."
        elif "NOT NULL" in msg:
            msg = "A required field is missing."
        return JSONResponse({"detail": msg}, status_code=409)

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/healthz")
    def healthz() -> dict:
        icm([1.0, 2.0], [1.0])  # a broken numpy import fails the check
        if store is not None:
            store.ping()  # so does a missing or unreadable volume
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

    @app.post("/api/table")
    def table(req: TableRequest) -> dict:
        return compute_table(req)

    @app.get("/api/payouts")
    def payouts(entrants: int = Query(ge=2, le=100_000), paid_pct: float = Query(default=12.0, gt=0, le=100)) -> dict:
        frac = payout_table(entrants, paid_pct / 100)
        return {"entrants": entrants, "places_paid": len(frac), "percent": [100 * x for x in frac]}

    # ---------- saved data: idempotent PUT/DELETE by phone-generated ids ----------

    def _put(table: str, row_id: str, body: Write) -> dict:
        return db().put(table, row_id, body.model_dump(exclude={"client_ts"}), body.client_ts)

    def _delete(table: str, row_id: str) -> Response:
        db().delete(table, row_id)
        return Response(status_code=204)

    @app.get("/api/tournaments")
    def tournaments() -> list[dict]:
        return db().tournaments()

    @app.put("/api/tournaments/{tid}")
    def put_tournament(t: TournamentIn, tid: RecordId) -> dict:
        return _put("tournaments", tid, t)

    @app.delete("/api/tournaments/{tid}", status_code=204)
    def delete_tournament(tid: RecordId) -> Response:
        return _delete("tournaments", tid)  # its spots stay, unfiled

    @app.get("/api/scenarios")
    def scenarios(tournament_id: str | None = None, unfiled: bool = False) -> list[dict]:
        if unfiled:
            return db().list("scenarios", tournament_id=None)
        if tournament_id:
            return db().list("scenarios", tournament_id=tournament_id)
        return db().list("scenarios")

    @app.put("/api/scenarios/{sid}")
    def put_scenario(sc: ScenarioIn, sid: RecordId) -> dict:
        return _put("scenarios", sid, sc)

    @app.delete("/api/scenarios/{sid}", status_code=204)
    def delete_scenario(sid: RecordId) -> Response:
        return _delete("scenarios", sid)

    @app.get("/api/players")
    def players() -> list[dict]:
        return db().list("players")

    @app.put("/api/players/{pid}")
    def put_player(pl: PlayerIn, pid: RecordId) -> dict:
        return _put("players", pid, pl)

    @app.delete("/api/players/{pid}", status_code=204)
    def delete_player(pid: RecordId) -> Response:
        return _delete("players", pid)  # and its observations

    @app.get("/api/observations")
    def observations(player_id: str | None = None, tournament_id: str | None = None) -> list[dict]:
        where = {k: v for k, v in (("player_id", player_id), ("tournament_id", tournament_id)) if v}
        return db().list("observations", **where)

    @app.put("/api/observations/{oid}")
    def put_observation(ob: ObservationIn, oid: RecordId) -> dict:
        return db().observe(oid, ob.player_id, ob.text, ob.seen_at, ob.tournament_id)

    @app.delete("/api/observations/{oid}", status_code=204)
    def delete_observation(oid: RecordId) -> Response:
        return _delete("observations", oid)

    @app.get("/api/export")
    def export() -> JSONResponse:
        return JSONResponse(
            db().export(), headers={"Content-Disposition": 'attachment; filename="deicm-export.json"'}
        )

    @app.post("/api/extract")
    def extract(image: UploadFile = File(...)) -> dict:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "Photo import is not configured on this server (ANTHROPIC_API_KEY is not set).")
        media_type = (image.content_type or "").lower()
        if media_type == "image/jpg":
            media_type = "image/jpeg"
        if media_type not in IMAGE_TYPES:
            raise HTTPException(422, "Upload a JPEG, PNG or WebP image.")
        data = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(422, "Image is larger than 8 MB.")
        if not data:
            raise HTTPException(422, "The image file is empty.")

        import anthropic

        try:
            state = extract_state(data, media_type)
        except RefusalError as e:
            raise HTTPException(422, f"The model declined to read this image: {e}") from e
        except anthropic.RateLimitError as e:
            log.warning("anthropic rate limit, request id %s", getattr(e, "request_id", None))
            raise HTTPException(502, "The vision service is busy; try again in a moment.") from e
        except anthropic.APIStatusError as e:
            log.error("anthropic status %s, request id %s", e.status_code, getattr(e, "request_id", None))
            raise HTTPException(502, "The vision service returned an error; try again.") from e
        except anthropic.APIConnectionError as e:
            log.error("anthropic connection error: %s", e)
            raise HTTPException(502, "Could not reach the vision service; try again.") from e
        return {"state": state.model_dump(), "prefill": derive_prefill(state)}

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
