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
import re


# All the different ways "Answer" is written in different languages.
# Inlined from nemo_rl.evals.answer_parsing to avoid a cross-package import.
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:\u200b\u200b\u200b\u200b\u200b\u200b",  # Korean invisible character
    r"\u0989\u09a4\u09cd\u09a4\u09b0\s*:",
    r"\u0909\u0924\u094d\u0924\u09b0\s*:",
    r"\u0989\u09a4\u09cd\u09a4\u09b0\u0983",
    r"\u0989\u09a4\u09cd\u09a4\u09b0\s*:",
    r"Antwort\s*:",
    r"\ub2f5\ubcc0\s*:",
    r"\uc815\ub2f5\s*:",
    r"\ub2f5\s*:",
    r"\u7b54\u6848\s*\uff1a",
    r"\u7b54\u6848\s*:",
    r"\u7b54\s*\uff1a",
    r"\u7b54\s*:",
    r"\u7b54\u590d\s*\uff1a",
    r"\u7b54\u66f0\s*\uff1a",
    r"\u0627\u0644\u0625\u062c\u0627\u0628\u0629:",
    r"\u0627\u0644\u062c\u0648\u0627\u0628:",
    r"\u0625\u062c\u0627\u0628\u0629:",
    r"\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0627\u0644\u0646\u0647\u0627\u0626\u064a\u0629:",
    r"\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0627\u0644\u0635\u062d\u064a\u062d\u0629:",
    r"\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0627\u0644\u0635\u062d\u064a\u062d\u0629 \u0647\u064a:",
    r"\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0647\u064a:",
    r"\u0627\u0644\u062c\u0648\u0627\u0628 \u0627\u0644\u0646\u0647\u0627\u0626\u064a:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"\u7b54\u3048\s*:",
    r"\u7b54\u3048\s*\uff1a",
    r"\u56de\u7b54\s*:",
    r"\u56de\u7b54\s*\uff1a",
    r"\u89e3\u7b54\s*:",
    r"Jawaban\s*:",
    r"R\u00e9ponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"\u00ccd\u00e1h\u00f9n\s*:",
    r"Id\u00e1h\u00f9n\s*:",
    r"A\u0300m\u1ecd\u0300na\u0300\s*:",
    r"\u00c0d\u00e1h\u00f9n\s*:",
    r"A\u0300nu\u0301go\u0323\s*:",
    r"\u00c0\u1e63\u00e0y\u00e0n\s*:",
]


_ANSWER_COLON_PATTERN = re.compile(
    rf"(?i)(?:{'|'.join(f'(?:{r})' for r in sorted(MULTILINGUAL_ANSWER_REGEXES, key=len, reverse=True))})[ \t]*",
)


def _last_answer_colon_string(string: str) -> str:
    """Extract the content after the last multilingual "Answer:"-style marker.

    Searches for the last occurrence of any multilingual "Answer:" pattern
    (as defined in MULTILINGUAL_ANSWER_REGEXES)
    and returns everything after it until the end of the string.

    Args:
        string: Input string to search.

    Returns:
        The content after the last "Answer:"-style marker, or "" if none found.
    """
    if not string:
        return ""
    matches = list(_ANSWER_COLON_PATTERN.finditer(string))
    if not matches:
        return ""
    return string[matches[-1].end():].strip()


def _last_boxed_string(string: str) -> str:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression (without the box) or empty string ("") if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx + 7: right_brace_idx].strip() if right_brace_idx is not None else ""


def extract_answer(string: str) -> str:
    """Extract answer from model output text.

    Tries \\boxed{} first, then multilingual ``Answer:`` as fallback.
    Returns the extracted string or ``""`` if nothing is found.

    Args:
        string: Model output text.

    Returns:
        The extracted answer string.
    """
    extracted_answer = _last_boxed_string(string)
    return extracted_answer if extracted_answer else _last_answer_colon_string(string)
