import json, random
import numpy as np
import tinker
import tinker.types as types
from practice import (
    format_prompt, process_example, sample_from_model,
    extract_sql, sql_matches, evaluate_test_set
)

# --- Setup ---
service_client = tinker.ServiceClient()
base_model = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=base_model)
tokenizer = training_client.get_tokenizer()

# Load a small set of examples to compare on
with open("sql_create_context_v4.json") as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
sample = data[:10]  # 10 in-distribution examples

# Novel schema examples from Step 7
novel = [
    {
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
        "answer": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
        "answer": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    {
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
        "answer": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
        "answer": "SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3",
    },
    {
        "context": "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)",
        "question": "How many students are enrolled in each department?",
        "answer": "SELECT c.department, COUNT(DISTINCT e.student_id) FROM courses c JOIN enrollments e ON c.id = e.course_id GROUP BY c.department",
    },
]

# --- Get base model ---
print("Saving base model weights...")
base_client = training_client.save_weights_and_get_sampling_client(name="compare-base")

# --- Train one epoch ---
print("\nPreparing training data...")
train_data = data[200:]
processed_train = [process_example(ex, tokenizer) for ex in train_data]
random.shuffle(processed_train)

BATCH_SIZE = 256
LEARNING_RATE = 5e-4
step = 0
print("Training...")
for batch_idx in range(0, len(processed_train), BATCH_SIZE):
    batch = processed_train[batch_idx: batch_idx + BATCH_SIZE]
    if not batch:
        break
    fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE))
    fwdbwd_result = fwdbwd_future.result()
    optim_future.result()
    to_arr = lambda x: x.to_numpy() if hasattr(x, "to_numpy") else np.array(x.tolist())
    logprobs = np.concatenate([to_arr(o["logprobs"]) for o in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([to_arr(d.loss_fn_inputs["weights"]) for d in batch])
    loss = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))
    step += 1
    if step % 100 == 0 or batch_idx + BATCH_SIZE >= len(processed_train):
        print(f"  Update {step}, loss: {loss:.4f}")

print("\nSaving fine-tuned weights...")
ft_client = training_client.save_weights_and_get_sampling_client(name="compare-finetuned")


def compare(label, examples):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, ex in enumerate(examples, 1):
        base_out = extract_sql(sample_from_model(base_client, tokenizer, ex["context"], ex["question"]))
        ft_out   = extract_sql(sample_from_model(ft_client,   tokenizer, ex["context"], ex["question"]))
        base_ok  = sql_matches(base_out, ex["answer"], schema=ex["context"])
        ft_ok    = sql_matches(ft_out,   ex["answer"], schema=ex["context"])

        print(f"\n[{i}] Q: {ex['question']}")
        print(f"    Schema:   {ex['context'][:80]}{'...' if len(ex['context']) > 80 else ''}")
        print(f"    Expected: {ex['answer']}")
        print(f"    Base:     {base_out}  {'✓' if base_ok else '✗'}")
        print(f"    Tuned:    {ft_out}  {'✓' if ft_ok else '✗'}")


compare("In-Distribution (from training dataset)", sample)
compare("Out-of-Distribution (novel schemas)", novel)

# Summary counts
def accuracy(client, examples):
    correct = sum(
        1 for ex in examples
        if sql_matches(extract_sql(sample_from_model(client, tokenizer, ex["context"], ex["question"])),
                       ex["answer"], schema=ex["context"])
    )
    return correct, len(examples)

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
b_in, n_in = accuracy(base_client, sample)
f_in, _    = accuracy(ft_client,   sample)
b_ood, n_ood = accuracy(base_client, novel)
f_ood, _     = accuracy(ft_client,   novel)
print(f"  In-distribution  — Base: {b_in}/{n_in}  Fine-tuned: {f_in}/{n_in}")
print(f"  Out-of-dist      — Base: {b_ood}/{n_ood}  Fine-tuned: {f_ood}/{n_ood}")
