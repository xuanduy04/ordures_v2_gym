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

import pytest

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming

from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    _build_chat_completions_payload,
    _build_judge_response,
    _extract_chat_completion_text,
    _messages_to_chat_format,
    _normalize_judge_server_url,
)


def _make_config(**overrides):
    base = dict(
        host="",
        port=0,
        entrypoint="",
        name="test",
        judge_server_url="0.0.0.0:8000",
        judge_model="test-model",
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    base.update(overrides)
    return LibraryJudgeMathResourcesServerConfig(**base)


class TestNormalizeJudgeServerUrl:
    def test_bare_host_port(self):
        assert _normalize_judge_server_url("0.0.0.0:8000") == "http://0.0.0.0:8000"

    def test_full_url(self):
        assert _normalize_judge_server_url("http://0.0.0.0:8000") == "http://0.0.0.0:8000"

    def test_drops_path(self):
        assert _normalize_judge_server_url("http://0.0.0.0:8000/extra/stuff/behind") == "http://0.0.0.0:8000"

    def test_https(self):
        assert _normalize_judge_server_url("https://judge.example.com:8443") == "https://judge.example.com:8443"

    def test_strips_whitespace_and_slashes(self):
        assert _normalize_judge_server_url("  0.0.0.0:8000/  ") == "http://0.0.0.0:8000"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _normalize_judge_server_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _normalize_judge_server_url("   ")


class TestMessagesToChatFormat:
    def test_string_content(self):
        msgs = [
            NeMoGymEasyInputMessage(role="system", content="be precise"),
            NeMoGymEasyInputMessage(role="user", content="judge this"),
        ]
        out = _messages_to_chat_format(msgs)
        assert out == [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "judge this"},
        ]

    def test_list_content_text_parts(self):
        msgs = [
            NeMoGymEasyInputMessage(
                role="user",
                content=[
                    {"type": "input_text", "text": "hello "},
                    {"type": "input_text", "text": "world"},
                ],
            ),
        ]
        out = _messages_to_chat_format(msgs)
        assert out == [{"role": "user", "content": "hello world"}]


class TestBuildChatCompletionsPayload:
    def test_maps_max_output_tokens_to_max_tokens(self):
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=8192, temperature=0.7, top_p=0.8
        )
        msgs = [NeMoGymEasyInputMessage(role="user", content="q")]
        payload = _build_chat_completions_payload(params, msgs, "my-model")
        assert payload["model"] == "my-model"
        assert payload["stream"] is False
        assert payload["max_tokens"] == 8192
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.8
        assert payload["messages"] == [{"role": "user", "content": "q"}]

    def test_omits_optional_when_none(self):
        params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        msgs = [NeMoGymEasyInputMessage(role="user", content="q")]
        payload = _build_chat_completions_payload(params, msgs, "m")
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert payload["model"] == "m"


class TestExtractChatCompletionText:
    def test_string_content(self):
        data = {"choices": [{"message": {"content": "Analysis... [[A=B]]"}}]}
        assert _extract_chat_completion_text(data) == "Analysis... [[A=B]]"

    def test_list_content(self):
        data = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
        assert _extract_chat_completion_text(data) == "ab"

    def test_no_choices(self):
        assert _extract_chat_completion_text({}) == ""
        assert _extract_chat_completion_text({"choices": []}) == ""


class TestBuildJudgeResponse:
    def test_wraps_judge_text_in_minimal_response(self):
        resp = _build_judge_response("[[A=B]]", "my-model")
        assert resp.model == "my-model"
        assert resp.id == "chat_completion_judge"
        assert resp.object == "response"
        assert resp.parallel_tool_calls is False
        assert resp.tool_choice == "none"
        assert resp.tools == []
        assert len(resp.output) == 1
        msg = resp.output[0]
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        assert len(msg.content) == 1
        assert msg.content[0].type == "output_text"
        assert msg.content[0].text == "[[A=B]]"


class TestConfigRequiredFields:
    def test_missing_judge_server_url_raises(self):
        with pytest.raises(Exception):
            _make_config(judge_server_url=None)

    def test_missing_judge_model_raises(self):
        with pytest.raises(Exception):
            _make_config(judge_model=None)

    def test_missing_judge_responses_create_params_raises(self):
        with pytest.raises(Exception):
            _make_config(judge_responses_create_params=None)

    def test_valid_config(self):
        cfg = _make_config()
        assert cfg.judge_server_url == "0.0.0.0:8000"
        assert cfg.judge_model == "test-model"

    def test_default_name(self):
        cfg = LibraryJudgeMathResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            judge_server_url="0.0.0.0:8000",
            judge_model="m",
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "math_with_judge"

    def test_no_judge_model_server_field(self):
        # This config must not carry the Gym-managed judge_model_server.
        cfg = _make_config()
        assert not hasattr(cfg, "judge_model_server") or "judge_model_server" not in cfg.model_fields


class TestInheritedLibraryVerifier:
    """Library verifier logic is inherited unchanged from the original LibraryJudgeMathResourcesServer."""

    def test_verify_answer_with_library_inherited(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        reward, extracted = server._verify_answer_with_library("4", "2 + 2 = \\boxed{4}")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

        reward, extracted = server._verify_answer_with_library("\\boxed{12}", "3 * 4 = 13")
        assert reward == pytest.approx(0.0)
        assert extracted == "13"

    def test_strip_math_delimiters_inherited(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        assert server._strip_math_delimiters("\\(x + 1\\)") == "x + 1"
        assert server._strip_math_delimiters("$x + 1$") == "x + 1"
        assert server._strip_math_delimiters("x + 1") == "x + 1"
