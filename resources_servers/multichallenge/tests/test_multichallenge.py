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

from resources_servers.multichallenge_original.app import AggregationMode, RubricEvaluation
from resources_servers.multichallenge.app import (
    MultiChallengeConfig,
    MultiChallengeServer,
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
    return MultiChallengeConfig(**base)


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
        cfg = MultiChallengeConfig(
            host="",
            port=0,
            entrypoint="",
            judge_server_url="0.0.0.0:8000",
            judge_model="m",
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        assert cfg.name == "multichallenge"

    def test_no_judge_model_server_field(self):
        # This config must not carry the Gym-managed judge_model_server.
        cfg = _make_config()
        assert not hasattr(cfg, "judge_model_server") or "judge_model_server" not in cfg.model_fields


class TestInheritedAggregation:
    """Aggregation logic is inherited unchanged from the original MultiChallengeServer."""

    def create_evaluations(self, scores: list[float]) -> list[RubricEvaluation]:
        return [
            RubricEvaluation(
                question=f"Q{i}",
                pass_criteria="YES",
                judge_prompt="...",
                judge_response="...",
                verdict="YES" if s >= 0.99 else "NO",
                score=s,
                weight=1.0,
            )
            for i, s in enumerate(scores)
        ]

    def test_aggregation_mean_inherited(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        cfg.aggregation_mode = AggregationMode.MEAN
        mock_client = MagicMock(spec=ServerClient)
        server = MultiChallengeServer.model_construct(config=cfg, server_client=mock_client)
        evaluations = self.create_evaluations([1.0, 0.5, 0.0])
        assert server._aggregate_scores(evaluations) == pytest.approx(0.5)

    def test_aggregation_all_inherited(self):
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient

        cfg = _make_config()
        cfg.aggregation_mode = AggregationMode.ALL
        mock_client = MagicMock(spec=ServerClient)
        server = MultiChallengeServer.model_construct(config=cfg, server_client=mock_client)
        evaluations = self.create_evaluations([1.0, 0.5, 0.0])
        assert server._aggregate_scores(evaluations) == 0.0

    def test_aggregation_mode_is_enum_not_str(self):
        # Regression: the inherited verify() accesses self.config.aggregation_mode.value,
        # which raises AttributeError if aggregation_mode is a plain str instead of an
        # AggregationMode enum. Pydantic must coerce the YAML string into the enum.
        cfg = _make_config(aggregation_mode="mean")
        assert isinstance(cfg.aggregation_mode, AggregationMode)
        assert cfg.aggregation_mode.value == "mean"
