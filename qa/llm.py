"""LLM clients. Gemini (free tier) for real runs, FakeLLM for tests.

Every call is cached on disk (keyed by model + prompt), so re-running the evaluation
costs nothing and stays within free-tier limits. A simple rate limiter spaces calls
to stay under the requests-per-minute quota, and 429 errors are retried with backoff.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "llm_cache.jsonl"


class GeminiLLM:
    def __init__(self, model: str | None = None, rpm: int | None = None, cache: Path | None = CACHE):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv(ROOT / ".env")
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise SystemExit("GEMINI_API_KEY is empty: add your key to claims-copilot/.env")
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.min_gap = 60.0 / (rpm or int(os.environ.get("GEMINI_RPM", "12")))
        self._last = 0.0
        self.cache_path = cache
        self.cache: dict[str, dict] = {}
        if cache and cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        self.calls = 0

    def _key(self, system: str, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()

    def generate_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        """Return (parsed JSON, info). info has latency_ms, tokens, cached."""
        key = self._key(system, prompt)
        if key in self.cache:
            row = self.cache[key]
            return row["output"], {**row["info"], "cached": True}

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,                         # same question -> same answer
            response_mime_type="application/json",   # structured output, no free text to parse
            response_schema=schema,
        )
        for attempt in range(6):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                break
            except Exception as e:  # rate limit or transient server error -> back off and retry
                msg = str(e)
                if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")) and attempt < 5:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise
        self.calls += 1
        usage = getattr(resp, "usage_metadata", None)
        info = {
            "latency_ms": round((time.time() - t0) * 1000),
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "model": self.model,
        }
        output = json.loads(resp.text)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "output": output, "info": info}) + "\n")
        self.cache[key] = {"output": output, "info": info}
        return output, {**info, "cached": False}

    def _send(self, model: str, contents, config):
        """One API call with rate limiting and retries. Returns (response, latency_ms)."""
        for attempt in range(6):
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                return resp, round((time.time() - t0) * 1000)
            except Exception as e:
                msg = str(e)
                if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")) and attempt < 5:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                raise

    def agent_turn(self, model: str, system: str, history: list[dict], tools: list[dict],
                   force: list[str] | None = None) -> tuple[dict, dict]:
        """One step of a tool-using agent.

        `history` is a list of serialized Gemini Content dicts (kept verbatim, including any
        thought signatures, so multi-turn function calling replays correctly). Returns
        (model Content as dict, info). Cached by the full history, like generate_json.
        """
        key = hashlib.sha256(json.dumps([model, system, tools, history, force], sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            row = self.cache[key]
            return row["output"], {**row["info"], "cached": True}

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"]) for t in tools])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            # ANY = the model must call a tool every turn; it ends by calling final_answer.
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=force)),
        )
        contents = [types.Content.model_validate(h) for h in history]
        resp, latency = self._send(model, contents, config)
        self.calls += 1
        usage = getattr(resp, "usage_metadata", None)
        info = {"latency_ms": latency, "model": model,
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None)}
        output = resp.candidates[0].content.model_dump(mode="json", exclude_none=True)
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "output": output, "info": info}) + "\n")
        self.cache[key] = {"output": output, "info": info}
        return output, {**info, "cached": False}


class FakeLLM:
    """Test double: `respond(prompt) -> dict` decides the JSON the 'model' returns."""

    model = "fake"

    def __init__(self, respond: Callable[[str], dict]):
        self.respond = respond
        self.prompts: list[str] = []

    def generate_json(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        self.prompts.append(prompt)
        return self.respond(prompt), {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0, "cached": False}
