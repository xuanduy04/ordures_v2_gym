# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared utilities for outsource Gym resource servers that connect to an
externally-hosted LLM judge via ``judge_server_url``.

Import these from new outsource ``app.py`` files instead of reimplementing
them locally.  The functions mirror the patterns in
``examples/custom_rewards/utils/llm_judge_utils.py`` and
``examples/custom_rewards/base_llm_judge.py``, but live inside the Gym tree
so they are importable by resource-server venvs.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3
from aiohttp import ClientTimeout

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_global_aiohttp_client

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------------------------------------------------------
# Shared constants (match base_llm_judge.py)
# ---------------------------------------------------------------------------
_RETRYABLE_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}
_MAX_RETRIES: int = 67
_WAIT_SECONDS_BEFORE_RETRY: float = 5
_REQUEST_TIMEOUT_SECONDS: int = 600
_MODELS_FETCH_MAX_ATTEMPTS: int = 6
_MODELS_FETCH_TIMEOUT: int = 10


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------
def _normalize_judge_server_url(judge_server_url: str, env_name: str = "") -> str:
    """Normalize a bare ``host:port`` (or full URL) to a ``scheme://netloc`` string.

    Accepts ``0.0.0.0:8000``, ``http://0.0.0.0:8000``,
    ``http://0.0.0.0:8000/extra/path`` (path is dropped).

    ``env_name`` is used only in error messages to identify the caller.
    """
    judge_server_url = judge_server_url.strip().strip("/")
    label = f"{env_name} config" if env_name else "config"
    if not judge_server_url:
        raise ValueError(f"{label} requires non-empty `judge_server_url`.")
    if "://" not in judge_server_url:
        judge_server_url = f"http://{judge_server_url}"
    parsed_url = urlparse(judge_server_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(
            f"Expected `judge_server_url` like '0.0.0.0:8000' or 'http://0.0.0.0:8000'."
        )
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


# ---------------------------------------------------------------------------
# Model-list fetch (for startup validation)
# ---------------------------------------------------------------------------
def _fetch_models_json_once(server_url: str) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Message-format conversion (Responses → Chat Completions)
# ---------------------------------------------------------------------------
def _messages_to_chat_format(messages: list[NeMoGymEasyInputMessage]) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
def _build_chat_completions_payload(
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    messages: list[NeMoGymEasyInputMessage],
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


# ---------------------------------------------------------------------------
# Response-text extraction
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Startup endpoint validation
# ---------------------------------------------------------------------------
def _validate_and_setup_judge_endpoint(
    env_name: str,
    judge_server_url: str,
    judge_model: str,
) -> str:
    """Normalize ``judge_server_url``, validate ``judge_model`` against
    ``/v1/models``, print confirmation, and return the normalized URL.

    Raises ``ValueError`` if the model is not available.
    """
    normalized = _normalize_judge_server_url(judge_server_url, env_name=env_name)
    models_url = f"{normalized}/v1/models"
    models_response = _fetch_models_json_once(models_url)
    if "status_code" in models_response:
        raise ValueError(
            f"Could not query '{models_url}'. Query attempt(s) returned: {models_response}"
        )
    available_models = [model.get("id", "") for model in models_response.get("data", [])]
    if judge_model not in available_models:
        raise ValueError(
            f"Judge model '{judge_model}' NOT found on server '{normalized}'. "
            f"Available models: {available_models}"
        )
    print(
        f"{env_name}: judge_server_url='{normalized}', judge_model='{judge_model}' "
        f"validated against /v1/models ({len(available_models)} model(s) available)."
    )
    return normalized


# ---------------------------------------------------------------------------
# Chat-completions POST with retry
# ---------------------------------------------------------------------------
async def _post_chat_completions(
    env_name: str,
    chat_completions_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST a chat-completions request to the external judge with retry on
    429/5xx.

    Returns the parsed JSON response, or ``{}`` on exhaustion (so verdict
    defaults to NOT-EQUAL / NO / score 0.0).
    """
    client = get_global_aiohttp_client()
    timeout = ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
    headers = {"Content-Type": "application/json"}
    attempt = 0
    while attempt < _MAX_RETRIES:
        attempt += 1
        try:
            async with client.post(chat_completions_url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status in _RETRYABLE_STATUS_CODES:
                    body = await response.text()
                    print(
                        f"[WARNING] {env_name} at attempt={attempt}: judge request "
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
                f"[WARNING] {env_name} at attempt={attempt}: judge request failed "
                f"with error ({exc}); retrying in {_WAIT_SECONDS_BEFORE_RETRY} second(s)..."
            )
            await asyncio.sleep(_WAIT_SECONDS_BEFORE_RETRY + random.uniform(0, 1))
    print(
        f"[WARNING] {env_name} at attempt={attempt}: Maximum retries reached; "
        "returning empty judge response (verdict will default to NOT EQUAL / NO / score 0.0)."
    )
    return {}
