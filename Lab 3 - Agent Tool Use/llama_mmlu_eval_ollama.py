"""
Llama 3.2-1B MMLU Evaluation Script (Ollama)

This script evaluates Llama 3.2-1B on a single MMLU subject using a local Ollama server.

Usage:
1. Install Ollama: https://ollama.com
2. Pull the model: ollama pull llama3.2:1b
3. Install deps: pip install datasets tqdm requests
4. Run: python llama_mmlu_eval.py
"""

import requests
import json
from datasets import load_dataset
from tqdm.auto import tqdm
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"

# Single MMLU subject to evaluate
MMLU_SUBJECT = "astronomy"


def check_ollama():
    """Verify the Ollama server is running and the model is available."""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.ConnectionError:
        print("ERROR: Cannot connect to Ollama server.")
        print(f"Make sure Ollama is running at {OLLAMA_BASE_URL}")
        raise SystemExit(1)

    models = [m["name"] for m in resp.json().get("models", [])]
    # Check if our model is available (handle tag variants like "llama3.2:1b" vs "llama3.2:1b-latest")
    model_found = any(MODEL_NAME in m for m in models)

    if model_found:
        print(f"Ollama server is running")
        print(f"Model '{MODEL_NAME}' is available")
    else:
        print(f"Model '{MODEL_NAME}' not found. Available models: {models}")
        print(f"Pull it with: ollama pull {MODEL_NAME}")
        raise SystemExit(1)

    print(f"Subject: {MMLU_SUBJECT}")
    print("=" * 70 + "\n")


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice."""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(prompt):
    """Get the model's prediction from Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 1,
                "temperature": 0.0,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    generated_text = resp.json().get("response", "").strip()

    answer = generated_text[:1].upper()
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"

    return answer


def evaluate_subject(subject):
    """Evaluate model on a specific MMLU subject."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'=' * 70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(prompt)

        if predicted_answer == correct_answer:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Result: {correct}/{total} correct = {accuracy:.2f}%")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("Llama 3.2-1B MMLU Evaluation (Ollama)")
    print("=" * 70 + "\n")

    check_ollama()

    start_time = datetime.now()
    result = evaluate_subject(MMLU_SUBJECT)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if not result:
        print("Evaluation failed.")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Subject: {MMLU_SUBJECT}")
    print(f"Total Questions: {result['total']}")
    print(f"Total Correct: {result['correct']}")
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Duration: {duration/60:.1f} minutes")
    print("=" * 70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"llama_3.2_1b_mmlu_{MMLU_SUBJECT}_{timestamp}.json"

    output_data = {
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "duration_seconds": duration,
        "subject_results": [result],
        "overall_accuracy": result["accuracy"],
        "total_correct": result["correct"],
        "total_questions": result["total"],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\nEvaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
