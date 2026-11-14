"""Demo CLI tests: real by default, mock only with --mock."""

import pytest

from agent_eligibility_claims import demo


def test_demo_without_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no .env file to load
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        demo.main([])


def test_demo_mock_runs_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert demo.main(["--mock"]) == 0
    out = capsys.readouterr().out
    assert "Loaded 4 tools" in out
    assert "Demo complete." in out


def test_demo_gemini_without_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no .env file to load
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        demo.main(["--provider", "gemini"])


def test_demo_gemini_flag_without_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        demo.main(["--gemini"])
