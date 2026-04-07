import json, random
import re, sqlite3
import tinker.types as types
import tinker
from collections import Counter
import numpy as np

def format_prompt(example: dict) -> tuple[str, str]:
    """Format example as prompt and completion for text-to-SQL."""
    prompt = f"""Table schema:
{example['context']}
Question: {example['question']}
SQL: """
    completion = example["answer"]
    return prompt, completion

def process_example(example: dict, tokenizer) -> types.Datum:
    """Convert a (question, context, answer) example into a Tinker Datum."""
    prompt, completion = format_prompt(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    # Add space before completion, end with \n\n so the model learns to stop
    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Next-token prediction: input is tokens[:-1], target is tokens[1:]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights": np.array(weights, dtype=np.float32),
        },
    )

def extract_sql(raw: str) -> str:
    """
    Extract SQL from model output. The base model often appends Answer:, Explanation:,
    or hallucinated query results. Take only the SQL part (before those markers).
    """
    if not raw:
        return ""
    for sep in ["\nAnswer:", "\nExplanation:", "\nSQL:"]:
        idx = raw.find(sep)
        if idx >= 0:
            raw = raw[:idx]
    return raw.strip()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison:
    - Strip <|end_of_text|> and similar special tokens
    - Treat single and double quotes as equivalent
    - Normalize whitespace
    """
    if not sql:
        return ""
    sql = re.sub(r"<\|[^|]+\|>", "", sql)
    sql = sql.strip()
    sql = sql.replace('"', "'")
    sql = " ".join(sql.split())
    sql = sql.rstrip(";")
    return sql


def _extract_literals(sql: str) -> tuple[list[str], list[float]]:
    """Extract string and numeric literals from a SQL query."""
    strings = re.findall(r"'([^']*)'", sql) + re.findall(r'"([^"]*)"', sql)
    numbers = [float(m.group(1)) for m in re.finditer(
        r"(?<![a-zA-Z_])(\d+(?:\.\d+)?)(?![a-zA-Z_])", sql
    ) if m.group(1)]
    return strings, numbers


def _build_db(schema: str, str_lits: list[str], num_lits: list[float],
              seed: int = 0, n_rows: int = 50) -> sqlite3.Connection:
    """Build an in-memory SQLite DB from schema, seeded with query literals."""
    rng = random.Random(seed)
    # Make text columns case-insensitive
    nocase = re.sub(
        r"\b(\w+(?:\(\d+\))?)\s*(?=,|\))",
        lambda m: m.group(0) + " COLLATE NOCASE"
        if any(k in m.group(1).upper() for k in ("VARCHAR", "TEXT", "CHAR"))
        else m.group(0),
        schema,
    )
    conn = sqlite3.connect(":memory:")
    for stmt in [s.strip() for s in nocase.split(";") if s.strip()]:
        if not stmt.upper().startswith("CREATE"):
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            continue
        hdr = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\((.+)\)", stmt,
                        re.IGNORECASE | re.DOTALL)
        if not hdr:
            continue
        table, cols = hdr.group(1), []
        for part in hdr.group(2).split(","):
            toks = part.strip().split()
            if len(toks) >= 2 and not toks[0].upper().startswith(
                ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")
            ):
                is_num = any(k in toks[1].upper() for k in
                             ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC"))
                pool = (list(set(num_lits)) + [float(i * 7 + len(cols) * 3 + 1) for i in range(8)]
                        if is_num else
                        list(set(str_lits)) + [f"{toks[0]}_v{i}" for i in range(6)])
                rng.shuffle(pool)
                cols.append(pool)
        for i in range(n_rows):
            vals = [p[i % len(p)] for p in cols]
            try:
                conn.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(vals))})", vals)
            except sqlite3.Error:
                break
    conn.commit()
    return conn


def sql_matches(generated: str, expected, schema: str = "") -> bool:
    """
    Check if generated SQL matches expected.
    With schema: execution-based (runs both on multiple seeded SQLite DBs).
    Without schema: normalized string comparison.
    """
    gen_sql = normalize_sql(extract_sql(generated))
    if not gen_sql:
        return False
    expected_list = [expected] if isinstance(expected, str) else expected

    if not schema:
        return any(gen_sql == normalize_sql(e) for e in expected_list)

    # Gather all literals from both sides for realistic seed data
    all_strs, all_nums = [], []
    for sql in [gen_sql] + [normalize_sql(e) for e in expected_list]:
        s, n = _extract_literals(sql)
        all_strs.extend(s)
        all_nums.extend(n)

    n_dbs = 5
    for exp in expected_list:
        exp_sql = normalize_sql(exp)
        if all(_exec_match(gen_sql, exp_sql, schema, all_strs, all_nums, seed=i * 97 + 13)
               for i in range(n_dbs)):
            return True
    return False


def _exec_match(gen_sql, exp_sql, schema, strs, nums, seed) -> bool:
    """Run both queries on one seeded DB and compare result multisets."""
    conn = _build_db(schema, strs, nums, seed=seed)
    try:
        gr = conn.execute(gen_sql).fetchall()
        er = conn.execute(exp_sql).fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    if not gr and not er:
        return True
    if gr and er and len(gr[0]) != len(er[0]):
        return False
    return Counter(gr) == Counter(er)


def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    """Generate SQL from the model given schema and question."""
    prompt = f"""Table schema:
{context}
Question: {question}
SQL: """
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    params = types.SamplingParams(max_tokens=150, temperature=0.0, stop=["\n\n", "Question:"])
    result = sampling_client.sample(prompt=model_input, sampling_params=params, num_samples=1).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""

def eval_one(sampling_client, tokenizer, ex: dict) -> bool:
    """Evaluate one example: generate SQL, then check if it matches expected."""
    sql = sample_from_model(sampling_client, tokenizer, ex["context"], ex["question"])
    return sql_matches(sql, ex["answer"], schema=ex["context"])

def evaluate_test_set(sampling_client, tokenizer, test_data: list) -> float:
    """Compute accuracy = fraction of test examples where generated SQL matches expected."""
    correct = sum(1 for ex in test_data if eval_one(sampling_client, tokenizer, ex))
    return correct / len(test_data)





with open("sql_create_context_v4.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"\nSample example:")
ex = data[10]
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")

NUM_TEST_EXAMPLES = 200  # Held-out for evaluation; all remaining data used for training
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]
print(f"Training examples: {len(train_data)} (all except evaluation)")
print(f"Test examples: {len(test_data)}")

print("Loading service client and tokenizer...")
service_client = tinker.ServiceClient()
base_model = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=base_model)
print("Processing training data into Tinker format...")
tokenizer = training_client.get_tokenizer()

print("\n--- Evaluating Base Model on 200 Test Questions ---")
base_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="base-model"
)
base_accuracy = evaluate_test_set(
    base_sampling_client, tokenizer, test_data
)
print(f"Base model accuracy: {base_accuracy:.2%} ({int(base_accuracy * 200)}/200)")



processed_train = [
    process_example(ex, tokenizer) for ex in train_data
]
random.shuffle(processed_train)


NUM_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 5e-4  # Tinker-recommended for Llama-3.2-1B with LoRA

step = 0
for epoch in range(NUM_EPOCHS):
    random.shuffle(processed_train)
    for batch_idx in range(0, len(processed_train), BATCH_SIZE):
        batch = processed_train[batch_idx : batch_idx + BATCH_SIZE]
        if len(batch) == 0:
            break

        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Compute loss (weighted cross-entropy over completion tokens only)
        to_arr = lambda x: x.to_numpy() if hasattr(x, "to_numpy") else np.array(x.tolist())
        logprobs = np.concatenate([to_arr(o["logprobs"]) for o in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([to_arr(d.loss_fn_inputs["weights"]) for d in batch])
        loss = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))

        step += 1
        if step % 100 == 0 or batch_idx + BATCH_SIZE >= len(processed_train):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, update {step}, loss: {loss:.4f}")
            
            
finetuned_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="finetuned"
)

finetuned_accuracy = evaluate_test_set(
    finetuned_sampling_client, tokenizer, test_data
)

print(f"Finetuned accuracy: {finetuned_accuracy:.2%}")


# --- Step 7: Test on Novel Schema Questions ---
print("\n--- Step 7: Novel Schema Questions (Out-of-Distribution) ---")

novel_questions = [
    # Easy
    {
        "difficulty": "Easy",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
        "expected": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "difficulty": "Easy",
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
        "expected": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    # Medium
    {
        "difficulty": "Medium",
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
        "expected": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "difficulty": "Medium",
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
        "expected": "SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3",
    },
    # Hard
    {
        "difficulty": "Hard",
        "context": "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)",
        "question": "How many students are enrolled in each department?",
        "expected": "SELECT c.department, COUNT(DISTINCT e.student_id) FROM courses c JOIN enrollments e ON c.id = e.course_id GROUP BY c.department",
    },
]

for i, ex in enumerate(novel_questions, 1):
    generated = sample_from_model(finetuned_sampling_client, tokenizer, ex["context"], ex["question"])
    correct = sql_matches(generated, ex["expected"], schema=ex["context"])
    print(f"\nQ{i} [{ex['difficulty']}]: {ex['question']}")
    print(f"  Generated: {extract_sql(generated)}")
    print(f"  Expected:  {ex['expected']}")
    print(f"  Match:     {'YES' if correct else 'NO'}")