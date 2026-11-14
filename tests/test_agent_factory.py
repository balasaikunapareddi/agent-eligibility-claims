"""Agent factory tests: real model by default, mock only when asked.

The default agent runs against real OpenAI and needs OPENAI_API_KEY. If
the key is missing, construction must fail with a clear message — never
silently fall back to the scripted model.
"""

import os

import pytest

from agent_eligibility_claims.agent import (
    build_scripted_agent,
    eligibility_agent,
    load_dotenv,
    require_openai_key,
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
