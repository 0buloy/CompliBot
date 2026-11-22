from complibot.cli import main

from conftest import FIXTURE_DATA


def test_cli_ingest_and_ask(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("COMPLIBOT_LLM_PROVIDER", "fake")
    base = ["--data-dir", str(FIXTURE_DATA), "--index-dir", str(tmp_path / "idx")]
    assert main(base + ["ingest"]) == 0
    out = capsys.readouterr().out
    assert "[internal] fake-tech-risk-notice" in out

    assert main(base + ["ask", "How many business days to stop using data after consent is withdrawn?"]) == 0
    out = capsys.readouterr().out
    assert "s.2.3" in out and "status=answered" in out

    assert main(base + ["search", "recovery time objective", "--role", "internal"]) == 0
    assert "fake-tech-risk-notice s.5.1.1" in capsys.readouterr().out


def test_cli_ask_without_index(tmp_path, capsys):
    assert main(["--index-dir", str(tmp_path / "missing"), "ask", "anything"]) == 1
    assert "complibot ingest" in capsys.readouterr().err
