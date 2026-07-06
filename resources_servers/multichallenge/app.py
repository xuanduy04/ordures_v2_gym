"""
MultiChallenge Environment Resources Server.

This is the production MultiChallenge server. The LLM-judge is NOT managed by
NeMo-Gym — the YAML config must supply a ``judge_server_url`` (a bare
``host:port``, e.g. ``0.0.0.0:8000``) pointing at an already-running baseline
``vllm serve`` endpoint, plus a ``judge_model`` name.

The judge is queried via the OpenAI **Chat Completions** API
(``{judge_server_url}/v1/chat/completions``) rather than the Responses API, since a
stock ``vllm serve`` endpoint only exposes the chat-completions surface.

All rubric / aggregation / verdict / request / response schema logic is
inherited unchanged from :class:`resources_servers.multichallenge_original.app.MultiChallengeServer`.
Any dataset that is compatible with ``multichallenge_simple_agent`` (the
original, Gym-managed-judge variant under ``multichallenge_original/``) works
here unchanged — the ``agent_ref.name`` is the same (``multichallenge_simple_agent``).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, List
from urllib.parse import urlparse

import requests
import urllib3
from aiohttp import ClientTimeout
from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_global_aiohttp_client

from resources_servers.multichallenge_original.app import (
    MultiChallengeServer as _OriginalMultiChallengeServer,
    RubricEvaluation,
    _extract_verdict,
)
from nemo_gym.base_resources_server import BaseResourcesServerConfig

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 67
_WAIT_SECONDS_BEFORE_RETRY = 5
_REQUEST_TIMEOUT_SECONDS = 600
_MODELS_FETCH_MAX_ATTEMPTS = 6
_MODELS_FETCH_TIMEOUT = 10


def _normalize_judge_server_url(judge_server_url: str) -> str:
    """Normalize a bare ``host:port`` (or full URL) to a ``scheme://netloc`` string.

    Accepts ``0.0.0.0:8000``, ``http://0.0.0.0:8000``,
    ``http://0.0.0.0:8000/extra/path`` (path is dropped).
    """
    judge_server_url = judge_server_url.strip().strip("/")
    if not judge_server_url:
        raise ValueError("multichallenge config requires non-empty `judge_server_url`.")
    if "://" not in judge_server_url:
        judge_server_url = f"http://{judge_server_url}"
    parsed_url = urlparse(judge_server_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(
            "Expected `judge_server_url` like '0.0.0.0:8000' or 'http://0.0.0.0:8000'."
        )
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def _fetch_models_json_once(server_url: str) -> dict:
    """Fetch ``/v1/models`` once with retries. Returns the parsed JSON, or a
    ``{"status_code": [...], "text": [...]}`` failure record."""
    responses: dict[str, list] = {"status_code": [], "text": []}
    attempt = 0
    while attempt < _MODELS_FETCH_MAX_ATTEMPTS:
        attempt += 1
        response = None
        try:
            response = requests.get(
                server_url,
                headers={"Accept": "application/json", "Connection": "close"},
                timeout=_MODELS_FETCH_TIMEOUT,
                stream=False,
                verify=False,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Querying '{server_url}', attempt {attempt}/{_MODELS_FETCH_MAX_ATTEMPTS}: {type(e).__name__}: {e}")
        finally:
            responses["status_code"].append(response.status_code if response is not None else None)
            responses["text"].append(response.text if response is not None else None)
            if response is not None:
                response.close()
    return responses


def _messages_to_chat_format(messages: List[NeMoGymEasyInputMessage]) -> list[dict[str, Any]]:
    """Convert NeMo-Gym easy input messages into OpenAI chat-completions ``messages``."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.content
        if isinstance(content, str):
            c = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    # Input text parts use type="input_text" (Responses API);
                    # be robust and key on the presence of a "text" field.
                    if "text" in part:
                        parts.append(str(part.get("text", "")))
                    elif "content" in part:
                        parts.append(str(part["content"]))
                else:
                    parts.append(str(part))
            c = "".join(parts)
        else:
            c = str(content)
        out.append({"role": m.role, "content": c})
    return out


def _build_chat_completions_payload(
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    messages: List[NeMoGymEasyInputMessage],
    judge_model: str,
) -> dict[str, Any]:
    """Build the JSON body for a ``/v1/chat/completions`` judge request.

    ``max_output_tokens`` (Responses-API naming) is mapped to ``max_tokens``.
    """
    payload: dict[str, Any] = {
        "model": judge_model,
        "messages": _messages_to_chat_format(messages),
        "stream": False,
    }
    if judge_responses_create_params.temperature is not None:
        payload["temperature"] = judge_responses_create_params.temperature
    if judge_responses_create_params.top_p is not None:
        payload["top_p"] = judge_responses_create_params.top_p
    if judge_responses_create_params.max_output_tokens is not None:
        payload["max_tokens"] = judge_responses_create_params.max_output_tokens
    return payload


def _extract_chat_completion_text(response_json: dict[str, Any]) -> str:
    """Extract the judge output text from a chat-completions response JSON."""
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


class MultiChallengeConfig(BaseResourcesServerConfig):
    """Configuration for the MultiChallenge environment server.

    The LLM-judge is hosted externally (not managed by NeMo-Gym). Both
    ``judge_server_url`` and ``judge_model`` are mandated (no defaults).
    """

    name: str = "multichallenge"

    # Bare host:port (or full URL) of an already-running vLLM endpoint.
    judge_server_url: str = Field(description="host:port of the externally-hosted LLM judge (e.g. 0.0.0.0:8000)")
    # Model name served at judge_server_url; validated against /v1/models at startup.
    judge_model: str = Field(description="Model name served at judge_server_url; sent as `model` in the judge payload")

    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests (max_output_tokens maps to max_tokens)"
    )

    aggregation_mode: Any = Field(default="mean", description="How to aggregate scores from multiple rubric items")
    judge_prompt_template: str = Field(
        default="""You are evaluating whether a model's response meets a specific criterion.

CONVERSATION CONTEXT:
{context}

MODEL'S FINAL RESPONSE:
{response}

EVALUATION QUESTION:
{question}

EXPECTED ANSWER: {pass_criteria}

Does the model's response satisfy the criterion described in the evaluation question?
Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line.""",
        description="Template for the judge evaluation prompt",
    )
    judge_system_message: str | None = Field(
        default="You are a precise evaluator. Assess responses objectively based on the given criteria.",
        description="Optional system message for the judge",
    )
    parallel_evaluation: bool = Field(default=True, description="Whether to evaluate rubric items in parallel")
    yes_label: str = Field(default="[[YES]]", description="Label indicating YES verdict")
    no_label: str = Field(default="[[NO]]", description="Label indicating NO verdict")


class MultiChallengeServer(_OriginalMultiChallengeServer):
    """MultiChallenge server that uses an externally-hosted LLM judge.

    Inherits all rubric / aggregation / verdict / verify logic from
    :class:`resources_servers.multichallenge_original.app.MultiChallengeServer`,
    overriding only the judge-call path to POST directly to
    ``{judge_server_url}/v1/chat/completions``.
    """

    config: MultiChallengeConfig

    # Derived in setup_webserver() from config.judge_server_url; not a YAML field.
    _judge_chat_completions_url: str = ""

    def setup_webserver(self):
        normalized = _normalize_judge_server_url(self.config.judge_server_url)
        self._judge_chat_completions_url = f"{normalized}/v1/chat/completions"

        models_url = f"{normalized}/v1/models"
        models_response = _fetch_models_json_once(models_url)
        if "status_code" in models_response:
            raise ValueError(
                f"Could not query '{models_url}'. Query attempt(s) returned: {models_response}"
            )
        available_models = [model.get("id", "") for model in models_response.get("data", [])]
        if self.config.judge_model not in available_models:
            raise ValueError(
                f"Judge model '{self.config.judge_model}' NOT found on server '{normalized}'. "
                f"Available models: {available_models}"
            )
        print(
            f"multichallenge: judge_server_url='{normalized}', judge_model='{self.config.judge_model}' "
            f"validated against /v1/models ({len(available_models)} model(s) available)."
        )
        return super().setup_webserver()

    async def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a chat-completions request to the external judge with retry on 429/5xx."""
        client = get_global_aiohttp_client()
        url = self._judge_chat_completions_url
        timeout = ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        headers = {"Content-Type": "application/json"}
        attempt = 0
        while attempt < _MAX_RETRIES:
            attempt += 1
            try:
                async with client.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status in _RETRYABLE_STATUS_CODES:
                        body = await response.text()
                        print(
                            f"[WARNING] multichallenge at attempt={attempt}: judge request "
                            f"returned status code ({response.status}) with body ({body[:300]}); "
                            f"retrying in {_WAIT_SECONDS_BEFORE_RETRY} second(s)..."
                        )
                        await asyncio.sleep(_WAIT_SECONDS_BEFORE_RETRY + random.uniform(0, 1))
                        continue
                    if response.status >= 400:
                        body = await response.text()
                        raise ValueError(
                            f"judge request failed with non-retryable status {response.status}: {body[:500]}"
                        )
                    return await response.json()
            except Exception as exc:
                print(
                    f"[WARNING] multichallenge at attempt={attempt}: judge request failed "
                    f"with error ({exc}); retrying in {_WAIT_SECONDS_BEFORE_RETRY} second(s)..."
                )
                await asyncio.sleep(_WAIT_SECONDS_BEFORE_RETRY + random.uniform(0, 1))
        print(
            f"[WARNING] multichallenge at attempt={attempt}: Maximum retries reached; "
            "returning empty judge response (verdict will default to NO / score 0.0)."
        )
        return {}

    async def _evaluate_rubric_item(self, item: dict, context: str, response: str) -> RubricEvaluation:
        """Evaluate a single rubric item using the externally-hosted LLM judge."""
        question = item.get("question", "")
        pass_criteria = item.get("pass_criteria", "YES")
        weight = item.get("weight", 1.0)

        judge_prompt = self.config.judge_prompt_template.format(
            context=context,
            response=response,
            question=question,
            pass_criteria=pass_criteria,
        )
        msgs: List[NeMoGymEasyInputMessage] = []
        if self.config.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=self.config.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=judge_prompt))

        payload = _build_chat_completions_payload(
            self.config.judge_responses_create_params, msgs, self.config.judge_model
        )
        response_json = await self._post_chat_completions(payload)
        judge_text = _extract_chat_completion_text(response_json)

        verdict = _extract_verdict(judge_text, self.config.yes_label, self.config.no_label)

        if pass_criteria.upper() == "YES":
            score = 1.0 if verdict == "YES" else 0.0
        elif pass_criteria.upper() == "NO":
            score = 1.0 if verdict == "NO" else 0.0
        else:
            score = 1.0 if verdict == "YES" else 0.0
        return RubricEvaluation(
            question=question,
            pass_criteria=pass_criteria,
            judge_prompt=judge_prompt,
            judge_response=judge_text,
            verdict=verdict,
            score=score,
            weight=weight,
        )


if __name__ == "__main__":
    MultiChallengeServer.run_webserver()
