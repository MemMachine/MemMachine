import importlib.util
import logging
import sys


def test_missing_spacy_is_not_logged_as_an_error(monkeypatch, caplog):
    # spaCy comes from the optional ``multihop`` dependency group, so a
    # missing install must not be reported as an error with a traceback.
    monkeypatch.setitem(sys.modules, "spacy", None)
    monkeypatch.delitem(
        sys.modules,
        "memmachine_server.retrieval_agent.agents.decomposer.decomposer",
        raising=False,
    )
    spec = importlib.util.find_spec(
        "memmachine_server.retrieval_agent.agents.rarag_query_agent"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with caplog.at_level(logging.INFO):
        spec.loader.exec_module(module)

    assert module.DECOMPOSER_AVAILABLE is False
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert any("multihop" in r.getMessage() for r in caplog.records)
