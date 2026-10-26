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
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from deicm.core import icm
from deicm.perturb import allin, bubble_factors

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
MAX_PLAYERS = 4000


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

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/healthz")
    def healthz() -> dict:
        icm([1.0, 2.0], [1.0])  # a broken numpy import fails the check
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

    @app.post("/api/table")
    def table(req: TableRequest) -> dict:
        return compute_table(req)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
