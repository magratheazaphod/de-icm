"""Command-line interface: ICM equity next to chip-EV equity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Percentages may overshoot 100 by this much (rounding of published payout tables).
PCT_TOLERANCE = 0.5


class CLIError(Exception):
    """Bad user input; reported on stderr with exit code 2."""


def icm(stacks, prizes):
    """Lazy wrapper so the CLI imports without the core and tests can patch it."""
    from deicm import icm as _icm

    return _icm(stacks, prizes)


def allin(stacks, prizes, winner, loser, pot=0.0):
    """Lazy wrapper around :func:`deicm.allin` (0-based seats), patchable in tests."""
    from deicm import allin as _allin

    return _allin(stacks, prizes, winner, loser, pot)


def bubble_factors(stacks, prizes, seats=None, pot=0.0):
    """Lazy wrapper around :func:`deicm.bubble_factors` (0-based seats), patchable in tests."""
    from deicm import bubble_factors as _bf

    return _bf(stacks, prizes, seats, pot)


def _parse_number(token: str, what: str) -> float:
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        raise CLIError(f"{what}: {token!r} is not a number") from None
    if not math.isfinite(value):
        raise CLIError(f"{what}: {token!r} is not finite")
    return value


def _read_numbers(path: str, what: str) -> list[float]:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise CLIError(f"cannot read {what} file {path!r}: {exc.strerror or exc}") from None
    values = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        values.extend(_parse_number(tok, f"{what} file {path!r}") for tok in line.split())
    if not values:
        raise CLIError(f"{what} file {path!r} contains no numbers")
    return values


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deicm",
        description="ICM equities and placement probabilities, shown next to chip-EV equity.",
        epilog="example: deicm 1000 2500 400 3000 --prizes 50 30 20",
    )
    p.add_argument("stacks", nargs="*", metavar="STACK", help="chip stacks, one per player")
    p.add_argument("--stacks-file", metavar="PATH",
                   help="file of stacks (whitespace or newline separated, # comments)")
    p.add_argument("--prizes", nargs="+", metavar="AMOUNT", help="prize amounts, 1st place first")
    p.add_argument("--prizes-file", metavar="PATH", help="file of prize amounts")
    p.add_argument("--prizes-pct", nargs="+", metavar="PCT",
                   help="prizes as percentages of the pool (needs --pool)")
    p.add_argument("--pool", metavar="AMOUNT", help="total prize pool for --prizes-pct")
    p.add_argument("--places", action="store_true",
                   help="also print the placement matrix for the paid places")
    p.add_argument("--allin", nargs=2, metavar=("W", "L"),
                   help="what-if: seat W wins an all-in against seat L (1-based seats)")
    p.add_argument("--bubble-factors", action="store_true",
                   help="print bubble factors and required equity between seats")
    p.add_argument("--seats", metavar="SPEC",
                   help="seats for --bubble-factors, 1-based, like 1-9, 1,3,5 or 1-4,7")
    p.add_argument("--pot", metavar="CHIPS",
                   help="dead money in the middle for --allin / --bubble-factors (default 0)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return p


def _resolve_stacks(args) -> list[float]:
    if args.stacks and args.stacks_file:
        raise CLIError("give stacks either as arguments or with --stacks-file, not both")
    if args.stacks_file:
        return _read_numbers(args.stacks_file, "stacks")
    if not args.stacks:
        raise CLIError("no stacks given (pass numbers or --stacks-file)")
    return [_parse_number(t, "stack") for t in args.stacks]


def _resolve_prizes(args) -> list[float]:
    sources = [args.prizes is not None, args.prizes_file is not None, args.prizes_pct is not None]
    if sum(sources) == 0:
        raise CLIError("no prizes given (use --prizes, --prizes-file or --prizes-pct with --pool)")
    if sum(sources) > 1:
        raise CLIError("use only one of --prizes, --prizes-file, --prizes-pct")
    if args.pool is not None and args.prizes_pct is None:
        raise CLIError("--pool is only used with --prizes-pct")
    if args.prizes is not None:
        return [_parse_number(t, "prize") for t in args.prizes]
    if args.prizes_file is not None:
        return _read_numbers(args.prizes_file, "prizes")
    if args.pool is None:
        raise CLIError("--prizes-pct needs --pool")
    pool = _parse_number(args.pool, "pool")
    if pool <= 0:
        raise CLIError("--pool must be positive")
    pcts = [_parse_number(t, "prize percentage") for t in args.prizes_pct]
    if any(x <= 0 for x in pcts):
        raise CLIError("prize percentages must be positive")
    if sum(pcts) > 100.0 + PCT_TOLERANCE:
        raise CLIError(f"prize percentages sum to {sum(pcts):g}, which exceeds 100")
    return [pool * x / 100.0 for x in pcts]


def _parse_seat(token: str, n: int, what: str) -> int:
    try:
        seat = int(token)
    except ValueError:
        raise CLIError(f"{what}: {token!r} is not a whole seat number") from None
    if not 1 <= seat <= n:
        raise CLIError(f"{what}: seat {seat} is out of range (1 to {n})")
    return seat


def _parse_seats(spec: str, n: int) -> list[int]:
    """Parse '1-9', '1,3,5' or '1-4,7' (1-based) into distinct 0-based seats, in order."""
    seats: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise CLIError(f"--seats: empty item in {spec!r}")
        if "-" in part:
            lo_s, sep, hi_s = part.partition("-")
            lo = _parse_seat(lo_s.strip(), n, "--seats")
            hi = _parse_seat(hi_s.strip(), n, "--seats")
            if lo > hi:
                raise CLIError(f"--seats: range {part!r} is backwards")
            seats.extend(range(lo, hi + 1))
        else:
            seats.append(_parse_seat(part, n, "--seats"))
    if len(set(seats)) != len(seats):
        raise CLIError(f"--seats: {spec!r} lists a seat more than once")
    return [x - 1 for x in seats]


def _json_float(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _json_matrix(a) -> list[list[float | None]]:
    return [[_json_float(v) for v in row] for row in np.asarray(a, dtype=float)]


def _allin_data(stacks, res, pot: float) -> dict:
    return {
        "pot": pot,
        "winner": int(res.winner) + 1,
        "loser": int(res.loser) + 1,
        "amount": float(res.amount),
        "busted": bool(res.busted),
        "stacks_before": [float(s) for s in stacks],
        "stacks_after": np.asarray(res.stacks_after, dtype=float).tolist(),
        "chip_ev_before": np.asarray(res.chip_ev_before, dtype=float).tolist(),
        "chip_ev_after": np.asarray(res.chip_ev_after, dtype=float).tolist(),
        "chip_delta": np.asarray(res.chip_delta, dtype=float).tolist(),
        "equity_before": np.asarray(res.equity_before, dtype=float).tolist(),
        "equity_after": np.asarray(res.equity_after, dtype=float).tolist(),
        "equity_fold": np.asarray(res.equity_fold, dtype=float).tolist(),
        "delta": np.asarray(res.delta, dtype=float).tolist(),
    }


def _bubble_data(res, pot: float) -> dict:
    return {
        "pot": pot,
        "seats": [int(x) + 1 for x in res.seats],
        "bf": _json_matrix(res.bf),
        "required_equity": _json_matrix(res.required_equity),
        "risk_premium": _json_matrix(res.risk_premium),
    }


def _render_allin(a: dict, dec: int) -> str:
    w, l = a["winner"], a["loser"]
    head = f"All-in: seat {w} wins {_stack_str(a['amount'])} chips from seat {l}"
    if a["pot"] > 0:
        head += f" plus a dead pot of {_stack_str(a['pot'])}"
    if a["busted"]:
        head += f"; seat {l} busts"
    header = ["seat", "stack", "stack after", "chipEV", "chipEV after", "chip delta",
              "ICM", "ICM after", "ICM delta"]
    rows = []
    for i in range(len(a["stacks_before"])):
        rows.append([
            str(i + 1),
            _stack_str(a["stacks_before"][i]),
            _stack_str(a["stacks_after"][i]),
            _fmt_num(a["chip_ev_before"][i], dec),
            _fmt_num(a["chip_ev_after"][i], dec),
            f"{a['chip_delta'][i]:+,.{dec}f}",
            _fmt_num(a["equity_before"][i], dec),
            _fmt_num(a["equity_after"][i], dec),
            f"{a['delta'][i]:+,.{dec}f}",
        ])
    wi, li = w - 1, l - 1
    lines = [head, ""] + _table(rows, header) + [
        "",
        f"seat {w} gains {a['delta'][wi]:,.{dec}f} in ICM versus {a['chip_delta'][wi]:,.{dec}f} in chip EV",
        f"seat {l} loses {-a['delta'][li]:,.{dec}f} in ICM versus {-a['chip_delta'][li]:,.{dec}f} in chip EV",
    ]
    return "\n".join(lines)


def _cell(v, fmt, diagonal: bool) -> str:
    if diagonal:
        return "-"
    return fmt(v) if v is not None else "n/a"


def _render_bubble(b: dict) -> str:
    seats = b["seats"]
    header = ["seat"] + [f"vs {s}" for s in seats]

    def block(matrix, fmt):
        rows = []
        for a, s in enumerate(seats):
            rows.append([str(s)] + [_cell(matrix[a][c], fmt, a == c) for c in range(len(seats))])
        return _table(rows, header)

    lines = ["Bubble factors (loss / gain; rows: seat at risk, columns: opponent):"]
    lines += block(b["bf"], lambda v: f"{v:.2f}")
    if b["pot"] > 0:
        pot_note = f"calling versus folding, {_stack_str(b['pot'])} chips dead in the pot"
    else:
        pot_note = "no dead money"
    lines += ["", f"Required equity (break-even win probability, {pot_note}):"]
    lines += block(b["required_equity"], lambda v: f"{100 * v:.1f}%")
    if b["pot"] > 0:
        lines += ["", f"Dead money in the pot: {_stack_str(b['pot'])} chips."]
    else:
        lines += ["", "Chip EV would give 1.00 and 50.0% everywhere."]
    return "\n".join(lines)


def _fmt_num(x: float, decimals: int) -> str:
    return f"{x:,.{decimals}f}"


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    widths = [max(len(header[c]), *(len(r[c]) for r in rows)) for c in range(len(header))]
    out = ["  ".join(h.rjust(w) for h, w in zip(header, widths))]
    out.append("  ".join("-" * w for w in widths))
    out.extend("  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in rows)
    return out


def _stack_str(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def _data(stacks, prizes, res) -> dict:
    placements = np.asarray(res.placements, dtype=float)
    chip_ev = np.asarray(res.chip_ev, dtype=float)
    delta = np.asarray(res.delta, dtype=float)
    h, M = res.grid
    return {
        "n_players": len(stacks),
        "places_paid": len(prizes),
        "pool": float(sum(prizes)),
        "grid": {"h": float(h), "M": int(M)},
        "stacks": [float(s) for s in stacks],
        "prizes": [float(p) for p in prizes],
        "chip_ev": chip_ev.tolist(),
        "equities": np.asarray(res.equities, dtype=float).tolist(),
        "delta": delta.tolist(),
        "delta_pct": (100.0 * delta / chip_ev).tolist(),
        "itm": np.asarray(res.itm, dtype=float).tolist(),
        "p_first": placements[:, 0].tolist(),
        "placements": placements.tolist(),
    }


def _render(d: dict, show_places: bool) -> str:
    dec = 2 if d["pool"] >= 100 else 4
    header = ["seat", "stack", "chipEV", "ICM", "delta", "delta%", "P(ITM)", "P(1st)"]
    rows = []
    for i in range(d["n_players"]):
        rows.append([
            str(i + 1),
            _stack_str(d["stacks"][i]),
            _fmt_num(d["chip_ev"][i], dec),
            _fmt_num(d["equities"][i], dec),
            f"{d['delta'][i]:+,.{dec}f}",
            f"{d['delta_pct'][i]:+.1f}%",
            f"{100 * d['itm'][i]:.1f}%",
            f"{100 * d['p_first'][i]:.1f}%",
        ])
    lines = _table(rows, header)
    if show_places:
        m = d["places_paid"]
        ph = ["seat"] + [f"{r}" for r in range(1, m + 1)]
        prow = [[str(i + 1)] + [f"{100 * q:.1f}%" for q in d["placements"][i]]
                for i in range(d["n_players"])]
        lines += ["", "Placement probabilities by place:"] + _table(prow, ph)
    h, M = d["grid"]["h"], d["grid"]["M"]
    lines += [
        "",
        f"players: {d['n_players']}   places paid: {d['places_paid']}   "
        f"total pool: {_fmt_num(d['pool'], dec)}",
        f"grid: h = {h:.6g} (1/{round(1 / h)}), M = {M}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed its message
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        stacks = _resolve_stacks(args)
        prizes = _resolve_prizes(args)
        n = len(stacks)
        if args.seats is not None and not args.bubble_factors:
            raise CLIError("--seats is only used with --bubble-factors")
        if args.pot is not None and args.allin is None and not args.bubble_factors:
            raise CLIError("--pot is only used with --allin or --bubble-factors")
        pot = 0.0
        if args.pot is not None:
            pot = _parse_number(args.pot, "--pot")
            if pot < 0:
                raise CLIError("--pot must not be negative")
        pair = None
        if args.allin is not None:
            w = _parse_seat(args.allin[0], n, "--allin")
            l = _parse_seat(args.allin[1], n, "--allin")
            if w == l:
                raise CLIError("--allin needs two different seats")
            pair = (w - 1, l - 1)
        seats = _parse_seats(args.seats, n) if args.seats is not None else None
        try:
            res = icm(stacks, prizes)
        except ValueError as exc:
            raise CLIError(str(exc)) from None
        data = _data(stacks, prizes, res)
        if pair is not None:
            try:
                data["allin"] = _allin_data(stacks, allin(stacks, prizes, *pair, pot=pot), pot)
            except ValueError as exc:
                raise CLIError(str(exc)) from None
        if args.bubble_factors:
            try:
                data["bubble_factors"] = _bubble_data(bubble_factors(stacks, prizes, seats, pot=pot), pot)
            except ValueError as exc:
                raise CLIError(str(exc)) from None
    except CLIError as exc:
        print(f"deicm: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        text = _render(data, args.places)
        dec = 2 if data["pool"] >= 100 else 4
        if "allin" in data:
            text += "\n\n" + _render_allin(data["allin"], dec)
        if "bubble_factors" in data:
            text += "\n\n" + _render_bubble(data["bubble_factors"])
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
