"""
Prompt templates and formatting for continual learning datasets.

Each task type has a template that converts raw examples into a
classification prompt. The ``format_example`` function is the main
entry point used by the dataset loader.
"""

from typing import Any, Dict, List, Optional

from .config import DatasetConfig


# ---------------------------------------------------------------------------
# Task-type prompt builders
# ---------------------------------------------------------------------------


def _format_sentiment(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format sentiment analysis tasks (sst2, yelp, amazon, imdb)."""
    text_field = config.text_fields[0]
    text = example.get(text_field, "")
    labels = ", ".join(config.label_map.values())
    return (
        f"Classify the sentiment of the following text.\n"
        f"Options: {labels}\n\n"
        f"Text: {text}\n"
        f"Sentiment:"
    )


def _format_topic(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format topic classification tasks (ag_news, dbpedia, yahoo)."""
    parts = []
    for field in config.text_fields:
        val = example.get(field, "")
        if val:
            parts.append(str(val))
    text = " ".join(parts)
    labels = ", ".join(config.label_map.values())
    return (
        f"Classify the topic of the following text.\n"
        f"Options: {labels}\n\n"
        f"Text: {text}\n"
        f"Topic:"
    )


def _format_nli(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format NLI tasks (mnli, rte, cb)."""
    premise = example.get(config.text_fields[0], "")
    hypothesis = example.get(config.text_fields[1], "")
    labels = ", ".join(config.label_map.values())
    return (
        f"Determine the relationship between the premise and hypothesis.\n"
        f"Options: {labels}\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Relationship:"
    )


def _format_paraphrase(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format paraphrase tasks (qqp)."""
    q1 = example.get(config.text_fields[0], "")
    q2 = example.get(config.text_fields[1], "")
    labels = ", ".join(config.label_map.values())
    return (
        f"Determine if the following two questions are paraphrases.\n"
        f"Options: {labels}\n\n"
        f"Question 1: {q1}\n"
        f"Question 2: {q2}\n"
        f"Answer:"
    )


def _format_qa(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format QA tasks (boolq, multirc)."""
    if config.name == "boolq":
        passage = example.get("passage", "")
        question = example.get("question", "")
        return (
            f"Answer the question based on the passage.\n"
            f"Options: true, false\n\n"
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    # multirc
    paragraph = example.get("paragraph", "")
    question = example.get("question", "")
    answer_text = example.get("answer", "")
    return (
        f"Is the answer correct based on the paragraph?\n"
        f"Options: true, false\n\n"
        f"Paragraph: {paragraph}\n"
        f"Question: {question}\n"
        f"Answer: {answer_text}\n"
        f"Correct:"
    )


def _format_causal(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format causal reasoning tasks (copa)."""
    premise = example.get("premise", "")
    choice1 = example.get("choice1", "")
    choice2 = example.get("choice2", "")
    return (
        f"Choose the more likely cause or effect.\n\n"
        f"Premise: {premise}\n"
        f"Choice 1: {choice1}\n"
        f"Choice 2: {choice2}\n"
        f"Answer:"
    )


def _format_wsd(example: Dict[str, Any], config: DatasetConfig) -> str:
    """Format word sense disambiguation tasks (wic)."""
    s1 = example.get("sentence1", "")
    s2 = example.get("sentence2", "")
    word = example.get("word", "")
    return (
        f"Does the word '{word}' have the same meaning in both sentences?\n"
        f"Options: true, false\n\n"
        f"Sentence 1: {s1}\n"
        f"Sentence 2: {s2}\n"
        f"Answer:"
    )




# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FORMATTERS = {
    "sentiment": _format_sentiment,
    "topic": _format_topic,
    "nli": _format_nli,
    "paraphrase": _format_paraphrase,
    "qa": _format_qa,
    "causal": _format_causal,
    "wsd": _format_wsd,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_example(
    example: Dict[str, Any],
    config: DatasetConfig,
    include_answer: bool = True,
) -> Dict[str, str]:
    """Format a dataset example into a prompt/answer pair.

    Args:
        example: Raw dataset example dict (must contain a ``label`` field).
        config: ``DatasetConfig`` for the task.
        include_answer: If True, the answer is appended to ``full_text``.
            Set to **False** for test data to prevent evaluation leakage.

    Returns:
        Dict with keys ``prompt``, ``answer``, ``full_text``.
    """
    formatter = _FORMATTERS.get(config.task_type)
    if formatter is None:
        raise ValueError(
            f"No prompt formatter for task_type={config.task_type!r}. "
            f"Available: {sorted(_FORMATTERS.keys())}"
        )

    prompt = formatter(example, config)
    label_idx = example.get("label", 0)
    answer = str(config.label_map.get(label_idx, str(label_idx)))

    if include_answer:
        full_text = f"{prompt} {answer}"
    else:
        full_text = prompt

    return {"prompt": prompt, "answer": answer, "full_text": full_text}


def create_chat_format(
    prompt: str,
    answer: str,
    system_message: str = "You are a helpful assistant.",
) -> List[Dict[str, str]]:
    """Wrap prompt/answer into an OpenAI-style chat message list.

    Returns:
        List of dicts with keys ``role`` and ``content``.
    """
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]