import re
import string
from collections import Counter

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def _basic_normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def exact_match_verifier(ground_truths: list[str], predictions: list[str]) -> float:
    best_score = 0.0

    for ground_truth in ground_truths:
        ground_truth_norm = _basic_normalize_answer(ground_truth)

        for prediction in predictions:
            pred_norm = _basic_normalize_answer(prediction)

            if pred_norm == ground_truth_norm:
                best_score = 1.0
                break

        if best_score == 1.0:
            break

    return best_score


def _normalize_text_based_answer(text: str | None) -> list[str]:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return text.split()


def F1_verifier(ground_truths: list[str], predictions: list[str]) -> float:
    best_score = 0.0

    for ground_truth in ground_truths:
        ground_truth_tokens = _normalize_text_based_answer(ground_truth)

        for prediction in predictions:
            pred_tokens = _normalize_text_based_answer(prediction)

            if not ground_truth_tokens and not pred_tokens:
                score = 1.0
            elif not ground_truth_tokens or not pred_tokens:
                score = 0.0
            else:
                common = Counter(ground_truth_tokens) & Counter(pred_tokens)
                num_same = sum(common.values())

                if num_same == 0:
                    score = 0.0
                else:
                    precision = num_same / len(pred_tokens)
                    recall = num_same / len(ground_truth_tokens)
                    score = 2 * precision * recall / (precision + recall)

            best_score = max(best_score, score)

    return best_score


def _strip_math_delimiters(s: str) -> str:
    """Strip outer math delimiters from input string.

    Many answer values are wrapped in \\(...\\) or $...$,
    which causes the parsers like ``math_verify`` to fail when we wrap them
    in \\boxed{}.  Removing these outer delimiters fixes parsing.
    """
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


# Use Latex and plain math extraction from predictions
# https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
_math_verify_metric = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    ),
)


def math_verify_verifier(ground_truths: list[str], predictions: list[str]) -> float:
    try:
        score, _ = _math_verify_metric(
            golds=["\\boxed{" + _strip_math_delimiters(ground_truth.strip()) + "}" 
                   for ground_truth in ground_truths],
            predictions=predictions,
        )
        return float(score)
    except ValueError:
        return 0.0 # unparsable ground_truths, most likely not math so return 0.0
