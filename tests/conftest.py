import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_collection_modifyitems(config, items):
    # Leakage tests are charter-mandated (CLAUDE.md "Leakage tests"). Until
    # Phase 3 implements them they are skipped loudly, never silently absent:
    # the terminal summary below prints how many remain unimplemented.
    for item in items:
        if item.get_closest_marker("leakage_stub"):
            item.add_marker(
                pytest.mark.skip(reason="LEAKAGE TEST NOT IMPLEMENTED (due Phase 3)")
            )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = terminalreporter.stats.get("skipped", [])
    n_stubs = sum(
        1 for rep in skipped if "LEAKAGE TEST NOT IMPLEMENTED" in str(rep.longrepr)
    )
    if n_stubs:
        terminalreporter.write_sep(
            "!",
            f"{n_stubs} charter-mandated leakage tests are NOT YET IMPLEMENTED "
            "(must reach 0 by end of Phase 3)",
        )
