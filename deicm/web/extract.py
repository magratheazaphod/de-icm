"""Tournament state from a photo or screenshot (docs/SPEC.md section 12).

``extract_state`` calls the Anthropic API (key from ``ANTHROPIC_API_KEY``).
``derive_prefill`` is pure and is what the tests exercise.
"""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import BaseModel

MODEL = "claude-opus-5-5"
SEATS_AT_TABLE = 9

INSTRUCTION = (
    "This is a photo or screenshot of a poker tournament clock, or of a tournament page "
    "(clock, structure, chips or payout tab). Extract the tournament state. Read only what is "
    "visible and never infer or compute missing numbers; use null for anything not shown. "
    "Give amounts in the currency shown, without symbols. If a payout row covers several "
    "places (for example 10th to 18th), return it as one entry with place_from and place_to. "
    "Put anything you could not read clearly in notes, and set confidence accordingly."
)


class Payout(BaseModel):
    place_from: int
    place_to: int
    amount: float


class TournamentState(BaseModel):
    source_kind: Literal["pokeratlas", "room_clock", "other"]
    venue: str | None = None
    event_name: str | None = None
    level: int | None = None
    small_blind: float | None = None
    big_blind: float | None = None
    ante: float | None = None
    ante_kind: Literal["per_player", "big_blind_ante", "none"] | None = None
    time_left_in_level_seconds: int | None = None
    entries: int | None = None
    players_remaining: int | None = None
    average_stack: float | None = None
    total_chips: float | None = None
    prize_pool: float | None = None
    places_paid: int | None = None
    payouts: list[Payout] = []
    confidence: Literal["high", "medium", "low"]
    notes: str = ""


class RefusalError(Exception):
    """The model declined to read the image."""


def extract_state(image_bytes: bytes, media_type: str) -> TournamentState:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(image_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ],
        output_format=TournamentState,
    )
    if response.stop_reason == "refusal":
        details = response.stop_details
        raise RefusalError((details.explanation if details else None) or "The model declined to read this image.")
    if response.parsed_output is None:
        raise ValueError("The model returned no readable tournament state; try a clearer image.")
    return response.parsed_output


def derive_prefill(state: TournamentState) -> dict:
    """Values the page can apply directly. Missing inputs give None, never a guess."""
    prizes: list[float] = []
    for p in sorted(state.payouts, key=lambda p: (p.place_from, p.place_to)):
        lo = max(p.place_from, 1)
        hi = min(max(p.place_to, lo), lo + 4000)
        prizes += [p.amount] * (hi - lo + 1)

    pot = None
    if state.small_blind is not None or state.big_blind is not None:
        ante = state.ante or 0.0
        if state.ante_kind == "per_player":
            ante_total = ante * SEATS_AT_TABLE
        elif state.ante_kind == "big_blind_ante":
            ante_total = ante
        else:
            ante_total = 0.0
        pot = (state.small_blind or 0.0) + (state.big_blind or 0.0) + ante_total

    others = None
    if state.players_remaining is not None and state.average_stack is not None:
        others = {"count": state.players_remaining - SEATS_AT_TABLE, "average": state.average_stack}

    return {
        "prizes": prizes or None,
        "pot": pot,
        "others": others,
        "players_remaining": state.players_remaining,
        "average_stack": state.average_stack,
    }
