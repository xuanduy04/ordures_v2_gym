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

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient

from resources_servers.general_qa.app import (
    GeneralQAResourcesServer,
    GeneralQAResourcesServerConfig,
    GeneralQAVerifyRequest,
    _build_judge_response,
)
from resources_servers.utils_qa.extract_answer import extract_answer
from resources_servers.utils_qa.verify_answer import (
    F1_verifier,
    exact_match_verifier,
    math_verify_verifier,
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
    return GeneralQAResourcesServerConfig(**base)


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
        cfg = GeneralQAResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            judge_server_url="0.0.0.0:8000",
            judge_model="m",
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "general_qa"

    def test_no_judge_model_server_field(self):
        cfg = _make_config()
        assert not hasattr(cfg, "judge_model_server") or "judge_model_server" not in cfg.model_fields

    def test_should_use_judge_default(self):
        cfg = _make_config()
        assert cfg.should_use_judge is False


class TestDeterministicVerifiers:
    def test_exact_match_verifier_match(self):
        score = exact_match_verifier(["Paris"], ["Paris"])
        assert score == 1.0

    def test_exact_match_verifier_case_insensitive(self):
        score = exact_match_verifier(["paris"], ["Paris"])
        assert score == 1.0

    def test_exact_match_verifier_whitespace(self):
        score = exact_match_verifier(["  Paris  "], ["Paris"])
        assert score == 1.0

    def test_exact_match_verifier_mismatch(self):
        score = exact_match_verifier(["Paris"], ["London"])
        assert score == 0.0

    def test_exact_match_verifier_multiple_ground_truths(self):
        score = exact_match_verifier(["Paris", "paris"], ["London"])
        assert score == 0.0
        score = exact_match_verifier(["Paris", "London"], ["London"])
        assert score == 1.0

    def test_math_verify_verifier_numeric(self):
        score = math_verify_verifier(["4"], ["4"])
        assert score == 1.0

    def test_math_verify_verifier_boxed(self):
        score = math_verify_verifier(["4"], ["\\boxed{4}"])
        assert score == 1.0

    def test_math_verify_verifier_mismatch(self):
        score = math_verify_verifier(["4"], ["5"])
        assert score == 0.0

    def test_math_verify_verifier_unparsable(self):
        score = math_verify_verifier(["not math"], ["something"])
        assert score == 0.0

    def test_F1_verifier_exact_match(self):
        score = F1_verifier(["the cat sat"], ["the cat sat"])
        assert score == 1.0

    def test_F1_verifier_partial_overlap(self):
        score = F1_verifier(["the cat sat on the mat"], ["the dog sat on the mat"])
        assert 0.0 < score < 1.0

    def test_F1_verifier_no_overlap(self):
        score = F1_verifier(["hello world"], ["foo bar"])
        assert score == 0.0

    def test_F1_verifier_both_empty(self):
        score = F1_verifier([""], [""])
        assert score == 1.0


class TestExtractAnswer:
    def test_extract_boxed(self):
        assert extract_answer("The answer is \\boxed{42}") == "42"

    def test_extract_last_boxed(self):
        assert extract_answer("\\boxed{wrong} \\boxed{correct}") == "correct"

    def test_extract_answer_colon(self):
        assert extract_answer("Answer: 42") == "42"

    def test_extract_boxed_priority(self):
        assert extract_answer("Answer: wrong \\boxed{correct}") == "correct"

    def test_extract_nothing(self):
        assert extract_answer("no answer here") == ""

    def test_extract_empty(self):
        assert extract_answer("") == ""


class TestVerifyAnswerDeterministically:
    def test_boxed_match(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("42", "The answer is \\boxed{42}")
        assert reward == 1.0
        assert extracted == "42"

    def test_answer_colon_match(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("Paris", "Answer: Paris")
        assert reward == 1.0
        assert extracted == "Paris"

    def test_exact_match(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("hello", "hello")
        assert reward == 1.0
        assert extracted == "hello"

    def test_mismatch(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("correct", "wrong")
        assert reward < 1.0
        assert extracted is not None

    def test_empty_strings(self):
        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = GeneralQAResourcesServer.model_construct(config=cfg, server_client=mock_client)
        server.model_post_init(None)

        reward, extracted = server._verify_answer_deterministically("", "")
        assert reward == 1.0
        assert extracted == ""
