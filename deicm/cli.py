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
        try:
            res = icm(stacks, prizes)
        except ValueError as exc:
            raise CLIError(str(exc)) from None
        data = _data(stacks, prizes, res)
    except CLIError as exc:
        print(f"deicm: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(_render(data, args.places))
    return 0


if __name__ == "__main__":
    sys.exit(main())
