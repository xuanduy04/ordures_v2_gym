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

from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _messages_to_chat_format,
    _normalize_judge_server_url,
)


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

    def test_env_name_in_error(self):
        with pytest.raises(ValueError, match="test_env"):
            _normalize_judge_server_url("", env_name="test_env")


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
        data = {"choices": [{"message": {"content": "Analysis... [[YES]]"}}]}
        assert _extract_chat_completion_text(data) == "Analysis... [[YES]]"

    def test_list_content(self):
        data = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
        assert _extract_chat_completion_text(data) == "ab"

    def test_no_choices(self):
        assert _extract_chat_completion_text({}) == ""
        assert _extract_chat_completion_text({"choices": []}) == ""
