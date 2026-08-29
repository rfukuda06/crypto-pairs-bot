# tests/test_selection_store.py
from pairsbot.core.types import PairSelection
from pairsbot.research.selection_store import load_selection, save_selection


def test_round_trip(tmp_path):
    path = str(tmp_path / "selection.json")
    assert load_selection(path) is None            # missing -> None
    sel = PairSelection(a="LTC", b="XLM", beta=0.298, pvalue=0.0217)
    save_selection(path, sel)
    got = load_selection(path)
    assert (got.a, got.b) == ("LTC", "XLM")
    assert got.beta == 0.298 and got.pvalue == 0.0217


def test_load_bad_file_returns_none(tmp_path):
    path = str(tmp_path / "selection.json")
    with open(path, "w") as f:
        f.write("not json")
    assert load_selection(path) is None
