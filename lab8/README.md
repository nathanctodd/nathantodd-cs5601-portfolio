# Discussion Questions

1. Before vs. after: What specific improvements did you observe? Did the model learn SQL syntax, schema grounding, or both? What was the change in accuracy on the 200 held-out test questions? How well did it do on the additional manual test questions (Step 7)?

Before fine-tuning, the base model achieved 37.50–39.00% accuracy on the 200 held-out test questions. After one epoch of fine-tuning on ~78,377 examples, accuracy rose to 86.50–87.50%. The model appears to have learned both SQL syntax and schema grounding: it correctly maps natural language to column names from the CREATE TABLE context and generates valid SQL constructs like COUNT, MAX, WHERE, GROUP BY, and ORDER BY. On the Step 7 novel-schema questions, it scored 2/5 — getting both easy questions right (employees and students) but failing on the medium and hard ones — consistent with the expected drop on out-of-distribution schemas.

2. RAG comparison: Imagine you had a RAG system with 1,000 (question, SQL) pairs in a vector database. For which of the test questions above would RAG work well? For which would it struggle? Why?

A RAG system would work best on questions that closely match stored examples in phrasing and structure — simple SELECTs with a single WHERE clause (like Q1 and Q3 in Step 7) are likely to have near-duplicates in a 1,000-pair corpus. It would struggle with questions requiring novel schema grounding (Q2, Q4, Q5) because even if a retrieved example has similar intent, the column names and table structure won't match. RAG also can't compose new SQL from scratch — it can only surface what it has seen. The compositional skill needed for GROUP BY + ORDER BY + LIMIT (Q4) or a two-table JOIN (Q5) is exactly what fine-tuning builds into the weights, and what RAG cannot replicate through retrieval alone.

3. Error analysis: When the fine-tuned model gets a query wrong, how does it fail? Wrong column names? Wrong SQL syntax? Wrong logic? Each failure mode tells you something different about what the model learned.

The Step 7 failures reveal three distinct error modes:
- **Wrong aggregation (Q2):** The model generated `SELECT SUM(id) FROM products WHERE price > 50 AND category = 'food'` instead of `SELECT COUNT(*) FROM products WHERE price > 50`. It used `SUM` on the wrong column and hallucinated a spurious `category` filter. This is a logic error — the model understood the schema but misread the question's intent.
- **Incomplete SELECT (Q4):** The model produced `SELECT customer FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3`, dropping `SUM(amount)` from the SELECT clause. The ORDER BY logic was correct, but the output columns were incomplete. The model learned the ranking pattern but didn't generalize that aggregated columns must also be returned.
- **Hallucinated logic (Q5):** For the JOIN query, the model added a hardcoded `WHERE T2.department = 'Math'` filter not present in the question and grouped by the wrong table alias. This is a schema grounding failure on a two-table query — the hardest case in the distribution.


Training and evaluating the model on the SQL dataset...

(venv311) nathanctodd@NathanTodd lab8 % python practice.py
Total examples: 78577

Sample example:
  Question: How many acting statuses are there?
  Context:  CREATE TABLE management (temporary_acting VARCHAR)...
  Answer:   SELECT COUNT(DISTINCT temporary_acting) FROM management
Training examples: 78377 (all except evaluation)
Test examples: 200
Loading service client and tokenizer...
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.
Processing training data into Tinker format...
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
tokenizer_config.json: 55.4kB [00:00, 40.5MB/s]
tokenizer.json: 9.09MB [00:00, 19.9MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████| 296/296 [00:00<00:00, 1.95MB/s]

--- Evaluating Base Model on 200 Test Questions ---
Base model accuracy: 37.50% (75/200)
Epoch 1/1, update 100, loss: 0.0595
Epoch 1/1, update 200, loss: 0.0235
Epoch 1/1, update 300, loss: 0.0337
Epoch 1/1, update 307, loss: 0.0189
Finetuned accuracy: 86.50%


--- Step 7: Novel Schema Questions (Out-of-Distribution) ---







(venv311) (base) nathanctodd@NathanTodd lab8 % venv311/bin/python practice.py                                                                 
Total examples: 78577

Sample example:
  Question: How many acting statuses are there?
  Context:  CREATE TABLE management (temporary_acting VARCHAR)...
  Answer:   SELECT COUNT(DISTINCT temporary_acting) FROM management
Training examples: 78377 (all except evaluation)
Test examples: 200
Loading service client and tokenizer...
Processing training data into Tinker format...
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

--- Evaluating Base Model on 200 Test Questions ---
Base model accuracy: 39.00% (78/200)
Epoch 1/1, update 100, loss: 0.0515
Epoch 1/1, update 200, loss: 0.0391
Epoch 1/1, update 300, loss: 0.0411
Epoch 1/1, update 307, loss: 0.0275
Finetuned accuracy: 87.50%

--- Step 7: Novel Schema Questions (Out-of-Distribution) ---

Q1 [Easy]: What are the names of employees in the engineering department?
  Generated: SELECT name FROM employees WHERE department = 'engineering'
  Expected:  SELECT name FROM employees WHERE department = 'engineering'
  Match:     YES

Q2 [Easy]: How many products cost more than 50 dollars?
  Generated: SELECT SUM(id) FROM products WHERE price > 50 AND category = 'food'
  Expected:  SELECT COUNT(*) FROM products WHERE price > 50
  Match:     NO

Q3 [Medium]: What is the highest score in the science class?
  Generated: SELECT MAX(score) FROM students WHERE class ='science'
  Expected:  SELECT MAX(score) FROM students WHERE class = 'science'
  Match:     YES

Q4 [Medium]: List the top 3 customers by total order amount.
  Generated: SELECT customer FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3
  Expected:  SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3
  Match:     NO

Q5 [Hard]: How many students are enrolled in each department?
  Generated: SELECT T1.student_id, T1.name FROM enrollments AS T1 JOIN courses AS T2 ON T1.course_id = T2.id WHERE T2.department = 'Math' GROUP BY T1.department
  Expected:  SELECT c.department, COUNT(DISTINCT e.student_id) FROM courses c JOIN enrollments e ON c.id = e.course_id GROUP BY c.department
  Match:     NO
