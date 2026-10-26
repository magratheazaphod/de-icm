import pytest

# (label -> [max abs placement error, max rel equity error, max |sum - 1|])
OBSERVED: dict[str, list[float]] = {}


def record(label: str, abs_err: float, rel_err: float, sums_err: float) -> None:
    cur = OBSERVED.setdefault(label, [0.0, 0.0, 0.0])
    cur[0] = max(cur[0], abs_err)
    cur[1] = max(cur[1], rel_err)
    cur[2] = max(cur[2], sums_err)


@pytest.fixture(scope="session", autouse=True)
def _noop():
    yield


def pytest_terminal_summary(terminalreporter):
    if not OBSERVED:
        return
    tr = terminalreporter
    tr.section("observed DE-ICM errors vs exact references")
    tr.write_line(f"{'group':<28}{'max abs P err':>16}{'max rel V err':>16}{'max |sum-1|':>16}")
    for k in sorted(OBSERVED):
        a, r, s = OBSERVED[k]
        tr.write_line(f"{k:<28}{a:>16.2e}{r:>16.2e}{s:>16.2e}")
    a = max(v[0] for v in OBSERVED.values())
    r = max(v[1] for v in OBSERVED.values())
    s = max(v[2] for v in OBSERVED.values())
    tr.write_line(f"{'OVERALL':<28}{a:>16.2e}{r:>16.2e}{s:>16.2e}")
    tr.write_line("paper: 1.1e-14 abs probability, 4.0e-14 relative equity")
