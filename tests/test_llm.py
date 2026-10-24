import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from pipeline import llm

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"],
          "additionalProperties": False}


class FakeClient:
    """Stands in for anthropic.Anthropic(): records every request, replies from a queue. No network."""

    def __init__(self, *replies):
        self.calls, self.replies = [], list(replies)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def ok(text='{"label": "storm_damage"}', stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
              stop_details=NS(category="cyber") if stop == "refusal" else None)


class Llm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_miss_calls_model_once_then_cache_hit(self):
        c = FakeClient(ok())
        a = llm.ask_json("classify this", SCHEMA, client=c, cache_dir=self.dir)
        b = llm.ask_json("classify this", SCHEMA, client=c, cache_dir=self.dir)
        self.assertEqual(a, {"label": "storm_damage"})
        self.assertEqual(a, b)
        self.assertEqual(len(c.calls), 1)  # second answer came from data/llm_cache
        self.assertEqual(len(list(self.dir.glob("*.json"))), 1)

    def test_cache_hit_needs_no_client_and_no_key(self):
        llm.ask_json("p", SCHEMA, client=FakeClient(ok()), cache_dir=self.dir)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm.ask_json("p", SCHEMA, cache_dir=self.dir), {"label": "storm_damage"})

    def test_cache_key_depends_on_model_prompt_system_and_schema(self):
        c = FakeClient(ok())
        llm.ask_json("p", SCHEMA, client=c, cache_dir=self.dir)
        llm.ask_json("p2", SCHEMA, client=c, cache_dir=self.dir)
        llm.ask_json("p", SCHEMA, system="s", client=c, cache_dir=self.dir)
        llm.ask_json("p", SCHEMA, client=c, cache_dir=self.dir, model="claude-sonnet-5-5")
        llm.ask_json("p", {**SCHEMA, "title": "x"}, client=c, cache_dir=self.dir)
        self.assertEqual(len(c.calls), 5)
        self.assertEqual(len(list(self.dir.glob("*.json"))), 5)

    def test_request_shape_for_claude_opus_5_5(self):
        c = FakeClient(ok())
        llm.ask_json("the claim text", SCHEMA, system="be exact", client=c, cache_dir=self.dir)
        kw = c.calls[0]
        self.assertEqual(llm.MODEL, "claude-opus-5-5")
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(kw["messages"], [{"role": "user", "content": "the claim text"}])
        self.assertEqual(kw["system"], "be exact")
        self.assertEqual(kw["output_config"]["format"], {"type": "json_schema", "schema": SCHEMA})
        self.assertEqual(kw["fallbacks"], "default")
        self.assertEqual(kw["betas"], ["server-side-fallback-2026-07-01"])
        self.assertGreaterEqual(kw["max_tokens"], 16000)
        for rejected in ("temperature", "top_p", "top_k", "thinking", "tool_choice", "budget_tokens"):
            self.assertNotIn(rejected, kw)  # all 400 on this model
        self.assertNotEqual(kw["messages"][-1]["role"], "assistant")  # no prefill

    def test_refusal_truncation_garbage_and_errors_become_llm_unavailable_and_are_not_cached(self):
        for reply in (ok(stop="refusal"), ok(stop="max_tokens"), ok(text="not json"), ok(text="[1]"),
                      RuntimeError("boom"), NS(stop_reason="end_turn", content=[])):
            with self.assertRaises(llm.LLMUnavailable, msg=str(reply)):
                llm.ask_json("p", SCHEMA, client=FakeClient(reply), cache_dir=self.dir)
        self.assertEqual(list(self.dir.glob("*.json")), [])

    def test_no_credentials_means_unavailable_never_a_guess(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(llm.LLMUnavailable) as cm:
                llm.ask_json("p", SCHEMA, cache_dir=self.dir)
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_secrets_and_claim_text_never_reach_the_cache(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-SECRET"}):
            llm.ask_json("CLAIM TEXT WITH NAME Jane Doe", SCHEMA, client=FakeClient(ok()), cache_dir=self.dir)
        body = next(self.dir.glob("*.json")).read_text(encoding="utf-8")
        self.assertNotIn("SECRET", body)
        self.assertNotIn("Jane Doe", body)
        self.assertEqual(json.loads(body)["response"], {"label": "storm_damage"})

    def test_source_has_no_hardcoded_key(self):
        src = Path(llm.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sk-ant-", src)


if __name__ == "__main__":
    unittest.main()
