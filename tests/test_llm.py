"""Tests for LLM client layer — local client, factory, parser, prompt manager."""
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.parsers import JSONOutputParser
from src.llm.prompt_manager import PromptManager
from src.llm.local_client import (
    LocalLLMClient,
    MODEL_REGISTRY,
    _format_prompt,
    _get_template_family,
)
from src.llm.client import LLMClient


# ── Template formatting ───────────────────────────────────────────────────────

def test_llama3_template_no_system():
    prompt = _format_prompt("Tell me about Bali.", "", "llama3", None)
    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "Tell me about Bali." in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt


def test_llama3_template_with_system():
    prompt = _format_prompt("Tell me about Bali.", "You are a tourism expert.", "llama3", None)
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "You are a tourism expert." in prompt
    assert "Tell me about Bali." in prompt


def test_qwen_template():
    prompt = _format_prompt("Plan a trip to Seoul.", "You are helpful.", "qwen", None)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "Plan a trip to Seoul." in prompt


def test_mistral_template():
    prompt = _format_prompt("Best food in Bangkok?", "", "mistral", None)
    assert "[INST]" in prompt
    assert "Best food in Bangkok?" in prompt
    assert "[/INST]" in prompt


def test_template_family_detection():
    assert _get_template_family("meta-llama/Meta-Llama-3-8B-Instruct") == "llama3"
    assert _get_template_family("Qwen/Qwen2.5-7B-Instruct") == "qwen"
    assert _get_template_family("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"
    assert _get_template_family("some-unknown-model") == "auto"


# ── Model registry ────────────────────────────────────────────────────────────

def test_registry_has_llama3():
    assert "llama3-8b" in MODEL_REGISTRY
    assert "llama3.1-8b" in MODEL_REGISTRY


def test_registry_has_open_models():
    assert "qwen2.5-7b" in MODEL_REGISTRY
    assert "mistral-7b" in MODEL_REGISTRY


def test_registry_fields():
    for key, entry in MODEL_REGISTRY.items():
        assert "hf_id" in entry
        assert "family" in entry
        assert "requires_token" in entry


def test_registry_open_models_no_token():
    assert not MODEL_REGISTRY["qwen2.5-7b"]["requires_token"]
    assert not MODEL_REGISTRY["mistral-7b"]["requires_token"]


def test_registry_llama_requires_token():
    assert MODEL_REGISTRY["llama3-8b"]["requires_token"]


# ── LocalLLMClient (without loading model) ────────────────────────────────────

def test_local_client_init():
    client = LocalLLMClient.__new__(LocalLLMClient)
    client.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    client.family = "llama3"
    client._loaded = False
    assert not client._loaded


def test_local_client_invalid_key():
    with pytest.raises(ValueError, match="Unknown model_key"):
        LocalLLMClient(model_key="nonexistent-model-xyz")


def test_local_client_custom_model_id():
    client = LocalLLMClient.__new__(LocalLLMClient)
    # Simulate what __init__ does for a custom model_id
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    client.model_id = model_id
    client.family = _get_template_family(model_id)
    assert client.family == "qwen"


# ── LLMClient factory ─────────────────────────────────────────────────────────

def test_factory_no_backend(monkeypatch):
    """With no env vars, factory should resolve to 'none'."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_KEY", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_ID", raising=False)
    client = LLMClient(backend="auto")
    assert client.backend == "none"
    assert not client.is_available


def test_factory_local_backend_explicit():
    """Explicit local backend sets backend name but doesn't load model yet."""
    with patch("src.llm.local_client.LocalLLMClient.__init__", return_value=None) as mock_init:
        # We test that LLMClient selects 'local' when backend='local'
        client = LLMClient.__new__(LLMClient)
        resolved = client._resolve_backend("local", None, "llama3-8b", None, None)
        assert resolved == "local"


def test_factory_anthropic_env(monkeypatch):
    """With ANTHROPIC_API_KEY set, auto should select anthropic."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.delenv("LOCAL_MODEL_KEY", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_ID", raising=False)
    client = LLMClient.__new__(LLMClient)
    resolved = client._resolve_backend("auto", None, "llama3-8b", None, None)
    assert resolved == "anthropic"


def test_factory_local_env_takes_priority(monkeypatch):
    """LOCAL_MODEL_ID in env should take priority over ANTHROPIC_API_KEY."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("LOCAL_MODEL_ID", "some-local-model")
    client = LLMClient.__new__(LLMClient)
    resolved = client._resolve_backend("auto", None, "llama3-8b", None, None)
    assert resolved == "local"


def test_factory_call_raises_when_no_backend(monkeypatch):
    """Calling without a backend should raise RuntimeError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_KEY", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_ID", raising=False)
    client = LLMClient(backend="auto")
    with pytest.raises(RuntimeError, match="No LLM backend configured"):
        client.call("test")


# ── JSON parser ───────────────────────────────────────────────────────────────

def test_parser_pure_json():
    parser = JSONOutputParser()
    obj, ok, err = parser.parse('[{"name": "Bali", "concept_class": "Island"}]')
    assert ok
    assert obj[0]["name"] == "Bali"


def test_parser_code_block():
    parser = JSONOutputParser()
    text = '```json\n[{"name": "Seoul", "concept_class": "City"}]\n```'
    obj, ok, _ = parser.parse(text)
    assert ok
    assert obj[0]["concept_class"] == "City"


def test_parser_embedded_json():
    parser = JSONOutputParser()
    text = 'Here are the concepts:\n[{"name": "Jeju", "concept_class": "Island"}]'
    obj, ok, _ = parser.parse(text)
    assert ok


def test_parser_trailing_comma_repair():
    parser = JSONOutputParser()
    text = '[{"name": "Bangkok", "concept_class": "City",}]'
    obj, ok, _ = parser.parse(text)
    assert ok
    assert obj[0]["name"] == "Bangkok"


def test_parser_list_wraps_dict():
    parser = JSONOutputParser()
    text = '{"concepts": [{"name": "Bali"}]}'
    lst, ok, _ = parser.parse_list(text)
    assert ok
    assert isinstance(lst, list)


def test_parser_invalid_json():
    parser = JSONOutputParser()
    obj, ok, err = parser.parse("this is not json at all !!!")
    assert not ok
    assert obj is None


def test_parser_truncated_array():
    """Truncated JSON array (missing closing ]) — should recover individual objects."""
    parser = JSONOutputParser()
    text = 'Here are the results:\n[{"subject": "Bali", "predicate": "located_in", "object": "Indonesia", "confidence": 0.9},\n{"subject": "Kuta", "predicate": "located_in", "object": "Bali", "confidence": 0.8},'
    lst, ok, _ = parser.parse_list(text)
    assert ok
    assert len(lst) == 2
    assert lst[0]["subject"] == "Bali"


# ── Prompt manager ────────────────────────────────────────────────────────────

def test_prompt_manager_render():
    pm = PromptManager(prompts_root=Path("prompts"))
    rendered = pm.render("Hello {{name}}, visit {{place}}!", {"name": "Ahmad", "place": "Bali"})
    assert rendered == "Hello Ahmad, visit Bali!"


def test_prompt_manager_missing_variable():
    pm = PromptManager(prompts_root=Path("prompts"))
    rendered = pm.render("Hello {{name}}!", {})
    assert "{{name}}" in rendered  # left as-is with warning


def test_prompt_manager_load_cot_template():
    pm = PromptManager(prompts_root=Path("prompts"))
    template = pm.load_template("concept_extraction", "cot")
    assert "{{text}}" in template
    assert "JSON" in template


def test_prompt_manager_list_templates():
    pm = PromptManager(prompts_root=Path("prompts"))
    templates = pm.list_templates()
    assert "concept_extraction" in templates
    assert "relation_extraction" in templates
    assert "repair" in templates
