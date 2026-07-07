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

from typing import Any, List

from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)

from resources_servers.multichallenge_original.app import (
    AggregationMode,
    MultiChallengeServer as _OriginalMultiChallengeServer,
    RubricEvaluation,
    _extract_verdict,
)
from nemo_gym.base_resources_server import BaseResourcesServerConfig

from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _post_chat_completions,
    _validate_and_setup_judge_endpoint,
)


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

    aggregation_mode: AggregationMode = Field(
        default=AggregationMode.MEAN, description="How to aggregate scores from multiple rubric items"
    )
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
        normalized = _validate_and_setup_judge_endpoint(
            "multichallenge", self.config.judge_server_url, self.config.judge_model
        )
        self._judge_chat_completions_url = f"{normalized}/v1/chat/completions"
        return super().setup_webserver()

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
        response_json = await _post_chat_completions(
            "multichallenge", self._judge_chat_completions_url, payload
        )
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
