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

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    _build_judge_response,
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


class TestLibraryVerifier:
    """Library verifier uses explicit boxed / answer-colon extraction before math_verify comparison."""

    def test_verify_answer_with_library_boxed(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        # Boxed answer matches expected
        reward, extracted = server._verify_answer_with_library("4", "2 + 2 = \\boxed{4}")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

        # Boxed answer matches expected (with latex in boxed)
        reward, extracted = server._verify_answer_with_library("\\boxed{12}", "3 * 4 = \\boxed{12}")
        assert reward == pytest.approx(1.0)
        assert extracted == "12"

        # Boxed answer matches expected (fraction)
        reward, extracted = server._verify_answer_with_library("4.0", "2 + 2 = \\boxed{\\frac{8}{2}}")
        assert reward == pytest.approx(1.0)
        assert extracted == "\\frac{8}{2}"

    def test_verify_answer_with_library_answer_colon(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        # Answer colon extraction fallback (no boxed)
        reward, extracted = server._verify_answer_with_library("4", "Answer: 4")
        assert reward == pytest.approx(1.0)
        assert extracted == "4"

        # Boxed takes priority over answer_colon when both present
        reward, extracted = server._verify_answer_with_library("3", "Answer: 5 \\boxed{3}")
        assert reward == pytest.approx(1.0)
        assert extracted == "3"

    def test_verify_answer_with_library_no_extraction(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        # No boxed and no answer_colon marker — extraction fails, falls back to
        # generated_answer, which does not match expected.
        reward, extracted = server._verify_answer_with_library("\\boxed{12}", "3 * 4 = 13")
        assert reward == pytest.approx(0.0)
        assert extracted == "3 * 4 = 13"

        # Empty strings — math_verify raises on empty box, caught by except path.
        reward, extracted = server._verify_answer_with_library("", "")
        assert reward == pytest.approx(0.0)
        assert extracted is None

    def test_verify_answer_with_library_exact(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        # Direct numeric comparison via boxed extraction
        reward, extracted = server._verify_answer_with_library("3", "\\boxed{3}")
        assert reward == pytest.approx(1.0)
        assert extracted == "3"

    def test_verify_answer_with_library_mismatch(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        # Boxed answer does not match expected
        reward, extracted = server._verify_answer_with_library("\\boxed{5}", "10 - 5 = \\boxed{4}")
        assert reward == pytest.approx(0.0)
        assert extracted == "4"

    def test_strip_math_delimiters(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = LibraryJudgeMathResourcesServer.model_construct(config=cfg, server_client=mock_client)

        assert server._strip_math_delimiters("\\(x + 1\\)") == "x + 1"
        assert server._strip_math_delimiters("$x + 1$") == "x + 1"
        assert server._strip_math_delimiters("x + 1") == "x + 1"
