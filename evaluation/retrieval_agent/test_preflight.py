from evaluation.retrieval_agent import preflight


def test_find_missing_modules_reports_unknown_modules():
    missing_modules = preflight.find_missing_modules(["sys", "definitely_missing_mod"])

    assert missing_modules == ["definitely_missing_mod"]


def test_preflight_main_succeeds_when_modules_exist(capsys):
    exit_code = preflight.main(["sys"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""


def test_preflight_main_prints_missing_modules(capsys):
    exit_code = preflight.main(["sys", "definitely_missing_mod"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "definitely_missing_mod" in captured.err
