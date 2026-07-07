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

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient

from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
)


def _make_config(**overrides):
    base = dict(
        host="",
        port=0,
        entrypoint="",
        name="test",
        genrm_server_url="0.0.0.0:8000",
        genrm_model="test-genrm-model",
        genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    base.update(overrides)
    return GenRMCompareConfig(**base)


class TestConfigRequiredFields:
    def test_missing_genrm_server_url_raises(self):
        with pytest.raises(Exception):
            _make_config(genrm_server_url=None)

    def test_missing_genrm_model_raises(self):
        with pytest.raises(Exception):
            _make_config(genrm_model=None)

    def test_missing_genrm_responses_create_params_raises(self):
        with pytest.raises(Exception):
            _make_config(genrm_responses_create_params=None)

    def test_valid_config(self):
        cfg = _make_config()
        assert cfg.genrm_server_url == "0.0.0.0:8000"
        assert cfg.genrm_model == "test-genrm-model"

    def test_default_name(self):
        cfg = GenRMCompareConfig(
            host="",
            port=0,
            entrypoint="",
            genrm_server_url="0.0.0.0:8000",
            genrm_model="m",
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "genrm_compare"

    def test_no_genrm_model_server_field(self):
        # This config must not carry the Gym-managed genrm_model_server.
        cfg = _make_config()
        assert not hasattr(cfg, "genrm_model_server") or "genrm_model_server" not in cfg.model_fields


class TestInheritedComparison:
    """Comparison logic is inherited unchanged from the original GenRMCompareResourcesServer."""

    def _make_response_obj(self, output_text: str) -> Dict[str, Any]:
        return {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ],
        }

    def _make_genrm_response(self, score_1: float, score_2: float, ranking: float) -> Dict[str, Any]:
        """Helper to create a mock GenRM chat-completions response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": f'{{"score_1": {score_1}, "score_2": {score_2}, "ranking": {ranking}}}'
                    }
                }
            ]
        }

    async def test_compare_single_response_returns_default(self) -> None:
        """Single response returns default score (no comparison possible)."""
        cfg = _make_config()
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer.model_construct(config=cfg, server_client=server_mock)

        req = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}],
            response_objs=[self._make_response_obj("Response 1")],
        )

        res = await rs.compare(req)

        assert len(res.rewards) == 1
        assert res.rewards[0] == pytest.approx(3.0)
        server_mock.post.assert_not_called()

    async def test_verify_returns_default(self) -> None:
        """Verify endpoint returns default score (stub implementation, inherited)."""
        from nemo_gym.base_resources_server import BaseVerifyRequest
        from nemo_gym.openai_utils import NeMoGymResponse

        cfg = _make_config()
        server_mock = MagicMock(spec=ServerClient)
        rs = GenRMCompareResourcesServer.model_construct(config=cfg, server_client=server_mock)

        req = BaseVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="resp",
                created_at=0.0,
                model="m",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
        )

        res = await rs.verify(req)
        assert res.reward == pytest.approx(3.0)

    def test_compare_three_responses_triggers_genrm_calls(self) -> None:
        """Three responses produce comparison tasks (circular: 3 pairs).
        We verify this by checking the comparison_pairs generation without
        actually running compare(), which would require mocking aiohttp.
        """
        from resources_servers.genrm_compare_original.utils import generate_comparison_pairs

        cfg = _make_config()
        assert cfg.comparison_strategy == "circular"

        pairs = generate_comparison_pairs(cfg.comparison_strategy, 3)
        assert len(pairs) == 3
        assert pairs == [(0, 1), (1, 2), (2, 0)]

    def test_default_config_values_match_original(self) -> None:
        """Outsource config defaults match the original genrm_compare config defaults."""
        cfg = _make_config()
        assert cfg.comparison_strategy == "circular"
        assert cfg.num_judges_per_comparison == 1
        assert cfg.aggregator_method == "simple_tiebreaker"
        assert cfg.reasoning_bonus == 0.0
        assert cfg.answer_bonus == 0.0
        assert cfg.default_score == 3.0
        assert cfg.default_ranking == 3.5
        assert cfg.use_principle is False
        assert cfg.reasoning_answer_repeat_penalty is True
        assert cfg.genrm_parse_retries == 3
        assert cfg.genrm_parse_retry_sleep_s == 0.2
