"""Agent factory tests: real model by default, mock only when asked.

The default agent runs against real OpenAI and needs OPENAI_API_KEY. If
the key is missing, construction must fail with a clear message — never
silently fall back to the scripted model.
"""

import os

import pytest

from agent_eligibility_claims.agent import (
    PROVIDERS,
    build_scripted_agent,
    eligibility_agent,
    load_dotenv,
    require_gemini_key,
    require_openai_key,
    require_provider_key,
)


def test_require_openai_key_fails_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no .env file to load
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        require_openai_key()


def test_default_agent_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no .env file to load
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        with eligibility_agent():
            pass


def test_load_dotenv_reads_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test-fake-key\n")
    try:
        load_dotenv(env_file)
        assert os.environ["OPENAI_API_KEY"] == "sk-test-fake-key"
    finally:
        # A leaked fake key would un-skip the live test — always clean up.
        os.environ.pop("OPENAI_API_KEY", None)


def test_load_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n")
    load_dotenv(env_file)
    assert os.environ["OPENAI_API_KEY"] == "sk-from-env"


def test_explicit_scripted_model_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with build_scripted_agent() as agent:
        assert agent is not None


def test_require_gemini_key_fails_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # empty dir: no .env file to load
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        require_gemini_key()


def test_require_gemini_key_points_at_ai_studio(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="aistudio.google.com"):
        require_provider_key("gemini")


def test_gemini_agent_builds_correct_model(tmp_path, monkeypatch):
    # Dummy key only — construction builds the model, it never calls it.
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.chdir(tmp_path)
    with eligibility_agent(provider="gemini") as agent:
        assert agent.model.config["model_id"] == "gemini/gemini-3.5-flash-lite"


def test_gemini_agent_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        with eligibility_agent(provider="gemini"):
            pass


def test_gemini_key_does_not_satisfy_openai_path(tmp_path, monkeypatch):
    # Each provider needs its own key — a Gemini key must not unlock OpenAI.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        with eligibility_agent(provider="openai"):
            pass


def test_unknown_provider_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Unknown provider"):
        require_provider_key("bogus")


def test_providers_table_lists_both_models():
    assert PROVIDERS["openai"]["model_id"] == "openai/gpt-4o-mini"
    assert PROVIDERS["gemini"]["model_id"] == "gemini/gemini-3.5-flash-lite"
    assert PROVIDERS["openai"]["key_env"] == "OPENAI_API_KEY"
    assert PROVIDERS["gemini"]["key_env"] == "GEMINI_API_KEY"
