# eval_memagent.py
import json
import requests
from pathlib import Path

MAX_INPUT_CHARS = 32000 * 4

def query_model(context, question, port=8000):
    prompt = f"{context}\n\nQuestion: {question}"
    # truncate from the end of context if too long, keeping the question intact
    if len(prompt) > MAX_INPUT_CHARS:
        question_part = f"\n\nQuestion: {question}"
        max_context_chars = MAX_INPUT_CHARS - len(question_part)
        context = context[:max_context_chars]
        prompt = f"{context}\n\nQuestion: {question}"

    response = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
        "model": "BytedTsinghua-SIA/RL-MemoryAgent-7B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
    })
    data = response.json()
    if "choices" not in data:
        print("API error:", data)
        return ""
    return data["choices"][0]["message"]["content"].strip()

def f1_score(prediction, ground_truths):
    pred_tokens = set(prediction.lower().split())
    best_f1 = 0
    for gt in ground_truths:
        gt_tokens = set(gt.lower().split())
        if not pred_tokens or not gt_tokens:
            continue
        common = pred_tokens & gt_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1

if __name__ == "__main__":
    for eval_file in sorted(Path(".").glob("eval_*.json")):
        with open(eval_file) as f:
            samples = json.load(f)
        
        scores = []
        for s in samples:
            pred = query_model(s["context"], s["input"])
            scores.append(f1_score(pred, s["answers"]))
        
        print(f"{eval_file.name}: F1={sum(scores)/len(scores):.3f} ({len(scores)} samples)")