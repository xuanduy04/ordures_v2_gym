"""
General QA Environment Resources Server.

This is a standalone (not inherited) QA resource server. It verifies model
responses against expected answers using three deterministic verifiers
(exact match, math-verify, F1) plus an optional externally-hosted LLM judge.

The LLM-judge is NOT managed by NeMo-Gym — the YAML config must supply a
``judge_server_url`` (host:port, e.g. ``0.0.0.0:8000``) pointing at an
already-running ``vllm serve`` endpoint, plus a ``judge_model`` name.

The judge is queried via ``{judge_server_url}/v1/chat/completions`` (native
vLLM), not the Responses API. The judge is only invoked when
``should_use_judge=true`` AND deterministic reward <= 0.5 (mixed-rewards
strategy).
"""

from __future__ import annotations

import contextlib
import logging
from io import StringIO
from typing import Any, Callable, ClassVar, List, Optional

from fastapi import FastAPI
from math_verify.errors import TimeoutException
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.utils_outsource.judge_server_url_utils import (
    _build_chat_completions_payload,
    _extract_chat_completion_text,
    _post_chat_completions,
    _validate_and_setup_judge_endpoint,
)
from resources_servers.utils_qa.extract_answer import extract_answer
from resources_servers.utils_qa.verify_answer import (
    F1_verifier,
    exact_match_verifier,
    math_verify_verifier,
)


class GeneralQARunRequest(BaseRunRequest):
    expected_answer: str
    should_use_judge: Optional[bool]


class GeneralQAVerifyRequest(GeneralQARunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class GeneralQAVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    deter_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]


class GeneralQAResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the GeneralQA environment server.

    The LLM-judge is hosted externally (not managed by NeMo-Gym). Both
    ``judge_server_url`` and ``judge_model`` are mandated (no defaults).
    """

    name: str = "general_qa"

    # Bare host:port (or full URL) of an already-running vLLM endpoint.
    judge_server_url: str = Field(description="host:port of the externally-hosted LLM judge (e.g. 0.0.0.0:8000)")
    # Model name served at judge_server_url; validated against /v1/models at startup.
    judge_model: str = Field(description="Model name served at judge_server_url; sent as `model` in the judge payload")

    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        description="Base parameters for judge model requests (max_output_tokens maps to max_tokens)"
    )

    should_use_judge: bool = False


def _build_judge_response(judge_text: str, judge_model: str) -> NeMoGymResponse:
    """Build a minimal NeMoGymResponse that wraps the chat-completions judge output
    for compatibility with the inherited ``JudgeEvaluation`` schema."""
    return NeMoGymResponse(
        id="chat_completion_judge",
        created_at=0.0,
        model=judge_model,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="chat_completion_judge_msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=judge_text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class GeneralQAResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to a problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only semantic or mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: GeneralQAResourcesServerConfig

    # Derived in setup_webserver() from config.judge_server_url; not a YAML field.
    _judge_chat_completions_url: str = ""

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        self._verifiers: list[Callable[[list[str], list[str]], float]] = [
            exact_match_verifier,
            math_verify_verifier,
            F1_verifier,
        ]

    def setup_webserver(self) -> FastAPI:
        normalized = _validate_and_setup_judge_endpoint(
            "general_qa", self.config.judge_server_url, self.config.judge_model
        )
        self._judge_chat_completions_url = f"{normalized}/v1/chat/completions"
        return super().setup_webserver()

    async def verify(self, body: GeneralQAVerifyRequest) -> GeneralQAVerifyResponse:
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        (
            reward,
            extracted_answer,
            deter_reward,
            judge_evaluations,
        ) = await self._verify_answer(body.expected_answer, combined_response, body.should_use_judge)
        
        return GeneralQAVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            deter_reward=deter_reward,
            judge_evaluations=judge_evaluations,
        )

    async def _verify_answer(
        self, expected_answer: str, generated_answer: str, should_use_judge: bool | None = None
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        in comparison with the specified expected answer.
        """

        deter_reward, extracted_answer = self._verify_answer_deterministically(expected_answer, generated_answer)
        
        # If the sample does not define whether a judge should be used, default back to config
        should_use_judge = self.config.should_use_judge if should_use_judge is None else should_use_judge
        if not should_use_judge or deter_reward > 0.5:
            return deter_reward, extracted_answer, deter_reward, None

        judge_answer = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(expected_answer, judge_answer)
        return judge_reward, extracted_answer, deter_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    def _verify_answer_deterministically(self, expected_answer: str, generated_answer: str) -> tuple[float, str | None]:
        """Verify the correctness of a generated answer using deterministic methods.
        """
        try:
            # try to manually parse the answer
            extracted = extract_answer(generated_answer)

            if not extracted:
                extracted = generated_answer  # default to generated_answer

            with self._mute_output():
                ret_score = max(verifier([expected_answer], [extracted]) 
                                for verifier in self._verifiers)

            return float(ret_score), extracted

        except (Exception, TimeoutException):
            return 0.0, None

    async def _verify_answer_with_judge(
        self, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        """Evaluate whether the two answers are equivalent using the externally-hosted LLM judge.

        Call ``{judge_server_url}/v1/chat/completions`` instead of the Gym-managed
        ``/v1/responses`` endpoint. Verdict parsing logic ([[A=B]] / [[A!=B]] label
        scanning) is identical to the original.
        """
        responses_create_params = self.config.judge_responses_create_params.model_copy(deep=True)

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            first_answer=first_answer, second_answer=second_answer
        )
        msgs: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(role="system", content=self.JUDGE_SYSTEM_MESSAGE),
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]
        responses_create_params.input = msgs

        payload = _build_chat_completions_payload(responses_create_params, msgs, self.config.judge_model)
        response_json = await _post_chat_completions(
            "general_qa", self._judge_chat_completions_url, payload
        )
        judge_text = _extract_chat_completion_text(response_json)

        judge_response = _build_judge_response(judge_text, self.config.judge_model)
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Verdict parsing identical to original: scan for [[A=B]] / [[A!=B]] labels.
        equal_choice_position = judge_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = judge_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation


if __name__ == "__main__":
    GeneralQAResourcesServer.run_webserver()
