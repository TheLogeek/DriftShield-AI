import json
from typing import Optional

import httpx


def evaluate_llm_output(
    prompt: str,
    generated_text: str,
    api_key: str,
    model: str = "gemini-2.0-flash",
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
) -> dict:
    judge_prompt = (
        "You are evaluating an LLM's generated output for quality issues. "
        "Assess whether the response contains:\n"
        "1. Hallucination — factual claims not supported by the prompt\n"
        "2. Format drift — response structure deviating from what was requested\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Generated output:\n{generated_text}\n\n"
        "Respond with a JSON object: "
        '{"hallucination": <0-1 score>, "format_drift": <0-1 score>, '
        '"explanation": "<brief reason>"}'
    )

    url = f"{api_base}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": judge_prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = httpx.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        return json.loads(text)
    except Exception as e:
        return {
            "hallucination": -1,
            "format_drift": -1,
            "explanation": f"Evaluation failed: {e}",
        }


def calibrate_judge(
    human_labeled: list[dict],
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    correct = 0
    total = len(human_labeled)
    for item in human_labeled:
        result = evaluate_llm_output(
            prompt=item["prompt"],
            generated_text=item["generated_text"],
            api_key=api_key,
            model=model,
        )
        judge_hallucinated = result.get("hallucination", 0) > 0.5
        human_hallucinated = item.get("hallucinated", False)
        if judge_hallucinated == human_hallucinated:
            correct += 1
    agreement = correct / total if total > 0 else 0.0
    return {
        "agreement_rate": agreement,
        "calibrated": agreement >= 0.7,
        "n_samples": total,
        "correct": correct,
    }
