"""CLI tests. The numerical core is replaced by a deterministic stub."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from deicm import cli


def fake_icm(stacks, prizes):
    s = np.asarray(stacks, dtype=float)
    p = np.asarray(prizes, dtype=float)
    if np.any(s <= 0):
        raise ValueError("all stacks must be positive")
    if len(p) > len(s):
        raise ValueError("more prizes than players")
    n, m = len(s), len(p)
    share = s / s.sum()
    # Deterministic, row-stochastic-ish placements: first place by share, rest even.
    q = np.empty((n, m))
    q[:, 0] = share
    if m > 1:
        q[:, 1:] = ((1 - share) / n)[:, None]
    equities = q @ p
    chip_ev = share * p.sum()
    return SimpleNamespace(placements=q, equities=equities, itm=q.sum(axis=1),
                           chip_ev=chip_ev, delta=equities - chip_ev, grid=(0.125, 250))


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    monkeypatch.setattr(cli, "icm", fake_icm)


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_positional_stacks_table(capsys):
    code, out, _ = run(capsys, "1000", "2500", "400", "3000", "--prizes", "50", "30", "20")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split() == ["seat", "stack", "chipEV", "ICM", "delta", "delta%", "P(ITM)", "P(1st)"]
    rows = [l.split() for l in lines[2:6]]
    assert [r[0] for r in rows] == ["1", "2", "3", "4"]
    assert [r[1] for r in rows] == ["1,000", "2,500", "400", "3,000"]
    assert rows[0][2] == "14.49"  # 1000/6900 * 100, pool >= 100 so 2 decimals
    assert "players: 4" in out and "places paid: 3" in out and "total pool: 100.00" in out
    assert "h = 0.125 (1/8), M = 250" in out


def test_columns_right_aligned(capsys):
    _, out, _ = run(capsys, "1", "10000", "--prizes", "60", "40")
    lines = out.splitlines()[:4]
    assert len({len(l) for l in lines}) == 1


def test_small_pool_uses_four_decimals(capsys):
    _, out, _ = run(capsys, "1", "1", "--prizes", "0.5", "0.25")
    assert "0.3750" in out
    assert "total pool: 0.7500" in out


def test_stacks_file_with_comments(tmp_path, capsys):
    f = tmp_path / "stacks.txt"
    f.write_text("# header\n1000 2500  # two\n400\n\n3000\n")
    code, out, _ = run(capsys, "--stacks-file", str(f), "--prizes", "50", "30", "20", "--json")
    assert code == 0
    assert json.loads(out)["stacks"] == [1000, 2500, 400, 3000]


def test_prizes_file(tmp_path, capsys):
    f = tmp_path / "p.txt"
    f.write_text("50\n30 # second\n20\n")
    _, out, _ = run(capsys, "5", "4", "3", "2", "--prizes-file", str(f), "--json")
    assert json.loads(out)["prizes"] == [50, 30, 20]


def test_prizes_pct_scaling(capsys):
    _, out, _ = run(capsys, "5", "4", "3", "2", "--prizes-pct", "50", "30", "20",
                    "--pool", "10000", "--json")
    d = json.loads(out)
    assert d["prizes"] == pytest.approx([5000, 3000, 2000])
    assert d["pool"] == pytest.approx(10000)


def test_prizes_pct_rounding_tolerated(capsys):
    code, _, _ = run(capsys, "5", "4", "3", "--prizes-pct", "50.2", "30.2", "20.0", "--pool", "100")
    assert code == 0


def test_prizes_pct_over_100(capsys):
    code, _, err = run(capsys, "5", "4", "3", "--prizes-pct", "60", "50", "--pool", "100")
    assert code == 2 and "exceeds 100" in err


def test_json_structure(capsys):
    _, out, _ = run(capsys, "1000", "2500", "400", "3000", "--prizes", "50", "30", "20", "--json")
    d = json.loads(out)
    assert d["n_players"] == 4 and d["places_paid"] == 3 and d["pool"] == 100
    assert d["grid"] == {"h": 0.125, "M": 250}
    for key in ("chip_ev", "equities", "delta", "delta_pct", "itm", "p_first", "stacks"):
        assert len(d[key]) == 4
    assert np.array(d["placements"]).shape == (4, 3)
    assert d["delta_pct"][0] == pytest.approx(100 * d["delta"][0] / d["chip_ev"][0])
    assert "seat" not in out  # no table


def test_places_matrix(capsys):
    _, out, _ = run(capsys, "1", "1", "1", "--prizes", "50", "30", "--places")
    assert "Placement probabilities by place" in out
    assert "33.3%" in out
    _, plain, _ = run(capsys, "1", "1", "1", "--prizes", "50", "30")
    assert "Placement probabilities" not in plain


@pytest.mark.parametrize("argv, msg", [
    ([], "no stacks"),
    (["1", "2"], "no prizes"),
    (["1", "abc", "--prizes", "5"], "not a number"),
    (["1", "2", "--prizes", "5", "--prizes-file", "x"], "only one of"),
    (["1", "2", "--prizes-pct", "50"], "needs --pool"),
    (["1", "2", "--prizes", "5", "--pool", "9"], "only used with"),
    (["1", "2", "--prizes-pct", "50", "--pool", "-3"], "positive"),
    (["--stacks-file", "/nonexistent/zzz", "--prizes", "5"], "cannot read"),
    (["1", "2", "--stacks-file", "/nonexistent/zzz", "--prizes", "5"], "not both"),
    (["1", "nan", "--prizes", "5"], "not finite"),
])
def test_input_errors(capsys, argv, msg):
    code, out, err = run(capsys, *argv)
    assert code == 2 and out == ""
    assert err.startswith("deicm: error:") and msg in err


def test_empty_file_error(tmp_path, capsys):
    f = tmp_path / "e.txt"
    f.write_text("# nothing\n")
    code, _, err = run(capsys, "--stacks-file", str(f), "--prizes", "5")
    assert code == 2 and "no numbers" in err


def test_icm_value_error_reported(capsys):
    code, _, err = run(capsys, "1", "-2", "--prizes", "5")
    assert code == 2 and "all stacks must be positive" in err
    code, _, err = run(capsys, "1", "2", "--prizes", "5", "4", "3")
    assert code == 2 and "more prizes than players" in err


def test_unknown_option_exit_code(capsys):
    assert cli.main(["--bogus"]) == 2


# ---- perturbation flags (stubbed numerical layer) ----

def fake_allin(stacks, prizes, winner, loser):
    n = len(stacks)
    a = float(min(stacks[winner], stacks[loser]))
    after = np.array(stacks, dtype=float)
    after[winner] += a
    after[loser] -= a
    busted = stacks[loser] <= stacks[winner]
    chip_before = np.array(stacks, dtype=float) / sum(stacks) * sum(prizes)
    chip_after = after / after.sum() * sum(prizes)
    eq_before = chip_before.copy()
    eq_after = chip_after.copy()
    eq_after[winner] -= 1.0
    eq_after[loser] += 0.5
    return SimpleNamespace(
        stacks_after=after, equity_before=eq_before, equity_after=eq_after, equity_fold=eq_before,
        delta=eq_after - eq_before, chip_ev_before=chip_before, chip_ev_after=chip_after,
        chip_delta=chip_after - chip_before, winner=winner, loser=loser, amount=a,
        busted=bool(busted))


def fake_bubble(stacks, prizes, seats=None):
    n = len(stacks)
    chosen = list(seats) if seats is not None else list(range(n if n <= 12 else 9))
    k = len(chosen)
    bf = np.full((k, k), 1.5)
    np.fill_diagonal(bf, np.nan)
    req = bf / (1 + bf)
    return SimpleNamespace(seats=tuple(chosen), bf=bf, required_equity=req,
                           risk_premium=req - 0.5)


@pytest.fixture
def pert(monkeypatch):
    calls = {}

    def a(stacks, prizes, w, l, pot=0.0):
        calls["allin"] = (w, l)
        calls["allin_pot"] = pot
        return fake_allin(stacks, prizes, w, l)

    def b(stacks, prizes, seats=None, pot=0.0):
        calls["bf"] = seats
        calls["bf_pot"] = pot
        return fake_bubble(stacks, prizes, seats)

    monkeypatch.setattr(cli, "allin", a)
    monkeypatch.setattr(cli, "bubble_factors", b)
    return calls


BASE = ["1000", "2500", "400", "3000", "--prizes", "50", "30", "20"]


def test_new_flags_absent_output_unchanged(capsys):
    _, out, _ = run(capsys, *BASE)
    assert "All-in" not in out and "Bubble" not in out


def test_allin_table(capsys, pert):
    code, out, _ = run(capsys, *BASE, "--allin", "4", "3")
    assert code == 0 and pert["allin"] == (3, 2)  # 1-based to 0-based
    assert "All-in: seat 4 wins 400 chips from seat 3; seat 3 busts" in out
    i = out.index("All-in")
    lines = out[i:].splitlines()
    assert lines[2].split() == ["seat", "stack", "stack", "after", "chipEV", "chipEV", "after",
                                "chip", "delta", "ICM", "ICM", "after", "ICM", "delta"]
    assert [l.split()[0] for l in lines[4:8]] == ["1", "2", "3", "4"]
    assert "seat 4 gains" in out and "seat 3 loses" in out
    assert "Bubble" not in out


def test_allin_no_bust_header(capsys, pert):
    _, out, _ = run(capsys, *BASE, "--allin", "3", "1")
    assert "seat 3 wins 400 chips from seat 1" in out and "busts" not in out


def test_bubble_tables(capsys, pert):
    code, out, _ = run(capsys, *BASE, "--bubble-factors", "--seats", "1-3")
    assert code == 0 and pert["bf"] == [0, 1, 2]
    assert "Bubble factors" in out and "Required equity" in out
    assert "1.50" in out and "60.0%" in out
    lines = out[out.index("Bubble factors"):].splitlines()
    assert lines[1].split() == ["seat", "vs", "1", "vs", "2", "vs", "3"]
    assert lines[3].split() == ["1", "-", "1.50", "1.50"]
    assert "Chip EV would give 1.00 and 50.0% everywhere." in out


def test_bubble_default_seats_passes_none(capsys, pert):
    run(capsys, *BASE, "--bubble-factors")
    assert pert["bf"] is None


def test_both_blocks_and_places(capsys, pert):
    code, out, _ = run(capsys, *BASE, "--allin", "1", "2", "--bubble-factors", "--places")
    assert code == 0
    assert "Placement probabilities" in out and "All-in" in out and "Bubble factors" in out
    assert out.index("All-in") < out.index("Bubble factors")


def test_json_keys(capsys, pert):
    _, out, _ = run(capsys, *BASE, "--allin", "4", "3", "--bubble-factors", "--seats", "2,4", "--json")
    d = json.loads(out)  # must be strict-parsable: NaN diagonals become null
    a = d["allin"]
    assert a["winner"] == 4 and a["loser"] == 3 and a["busted"] is True and a["amount"] == 400
    for key in ("stacks_before", "stacks_after", "chip_ev_before", "chip_ev_after",
                "chip_delta", "equity_before", "equity_after", "equity_fold", "delta"):
        assert len(a[key]) == 4
    b = d["bubble_factors"]
    assert b["seats"] == [2, 4]
    assert b["bf"][0][0] is None and b["bf"][0][1] == 1.5
    assert set(b) == {"pot", "seats", "bf", "required_equity", "risk_premium"}


def test_json_omits_keys_when_not_requested(capsys, pert):
    _, out, _ = run(capsys, *BASE, "--json")
    d = json.loads(out)
    assert "allin" not in d and "bubble_factors" not in d


@pytest.mark.parametrize("spec, expected", [
    ("1-9", list(range(9))), ("1,3,5", [0, 2, 4]), ("1-4,7", [0, 1, 2, 3, 6]),
    ("3", [2]), (" 2 , 4-5 ", [1, 3, 4]),
])
def test_parse_seats(spec, expected):
    assert cli._parse_seats(spec, 9) == expected


@pytest.mark.parametrize("spec, msg", [
    ("0-3", "out of range"), ("1-10", "out of range"), ("4-2", "backwards"),
    ("1,,2", "empty item"), ("a", "not a whole"), ("1,1", "more than once"),
    ("1-3,2", "more than once"), ("", "empty item"),
])
def test_parse_seats_errors(spec, msg):
    with pytest.raises(cli.CLIError, match=msg):
        cli._parse_seats(spec, 9)


@pytest.mark.parametrize("extra, msg", [
    (["--allin", "1", "1"], "different seats"),
    (["--allin", "0", "1"], "out of range"),
    (["--allin", "1", "5"], "out of range"),
    (["--allin", "x", "1"], "not a whole"),
    (["--seats", "1-2"], "only used with --bubble-factors"),
    (["--bubble-factors", "--seats", "1-9"], "out of range"),
    (["--bubble-factors", "--seats", "2,2"], "more than once"),
])
def test_perturbation_errors(capsys, pert, extra, msg):
    code, out, err = run(capsys, *BASE, *extra)
    assert code == 2 and out == ""
    assert err.startswith("deicm: error:") and msg in err


def test_library_value_error_from_perturbation(capsys, monkeypatch):
    def boom(*a, **k):
        raise ValueError("kaboom")
    monkeypatch.setattr(cli, "allin", boom)
    code, _, err = run(capsys, *BASE, "--allin", "1", "2")
    assert code == 2 and "kaboom" in err


def test_pot_flag_passed_and_in_json(capsys, pert):
    _, out, _ = run(capsys, *BASE, "--allin", "1", "2", "--bubble-factors", "--pot", "150", "--json")
    d = json.loads(out)
    assert pert["allin_pot"] == 150 and pert["bf_pot"] == 150
    assert d["allin"]["pot"] == 150 and d["bubble_factors"]["pot"] == 150
    _, out, _ = run(capsys, *BASE, "--allin", "1", "2", "--json")
    assert json.loads(out)["allin"]["pot"] == 0 and pert["allin_pot"] == 0.0


def test_pot_rendering(capsys, pert):
    _, out, _ = run(capsys, *BASE, "--allin", "3", "1", "--bubble-factors", "--pot", "150")
    assert "plus a dead pot of 150" in out and "Dead money in the pot: 150 chips." in out


@pytest.mark.parametrize("extra, msg", [
    (["--pot", "5"], "only used with"),
    (["--allin", "1", "2", "--pot", "-1"], "must not be negative"),
    (["--allin", "1", "2", "--pot", "x"], "not a number"),
])
def test_pot_errors(capsys, pert, extra, msg):
    code, out, err = run(capsys, *BASE, *extra)
    assert code == 2 and out == "" and msg in err
