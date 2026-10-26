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
