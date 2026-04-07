# Discussion Questions
1. Before vs. after: What specific improvements did you observe? Did the model learn SQL syntax, schema grounding, or both? What was the change in accuracy on the 200 held-out test questions? How well did it do on the additional manual test questions (Step 7)?
- Before the fine-tuning, the base model achieved an accuracy of 37.50% on the 200 held-out test questions. After fine-tuning, the accuracy improved significantly to 86.50%. This suggests that the model learned both SQL syntax and schema grounding, as it was able to generate correct SQL queries based on the provided context. The improvement in accuracy indicates that the model was able to better understand the structure of the database and how to formulate queries accordingly.



Hint: on the 200 in-distribution test questions, accuracy typically improves from ~37% (base) to ~87% (fine-tuned). The Step 7 questions use novel schemas and may show lower accuracy.

2. RAG comparison: Imagine you had a RAG system with 1,000 (question, SQL) pairs in a vector database. For which of the test questions above would RAG work well? For which would it struggle? Why?
- A RAG (Retrieval-Augmented Generation) system with 1,000 (question, SQL) pairs in a vector database would work well for test questions that are similar to the examples in the database. If a test question closely matches one of the stored pairs, the RAG system can retrieve the relevant SQL query and provide an accurate answer. However, it would struggle with test questions that are significantly different from the stored examples, especially those that require understanding of novel schemas or complex logic that is not represented in the database. The RAG system relies heavily on the quality and diversity of the stored examples, so it may fail to generalize to unseen questions or schemas.

3. Error analysis: When the fine-tuned model gets a query wrong, how does it fail? Wrong column names? Wrong SQL syntax? Wrong logic? Each failure mode tells you something different about what the model learned.
- When the fine-tuned model gets a query wrong, it can fail in several ways:
  - Wrong column names: This indicates that the model may not have fully learned the schema grounding or may have difficulty mapping the natural language question to the correct database schema.
  - Wrong SQL syntax: This suggests that the model may not have fully grasped the SQL syntax rules, which could lead to generating queries that are not executable.
  - Wrong logic: This implies that the model may understand the syntax and schema but fails to correctly interpret the question's intent, leading to incorrect query formulation.
Each failure mode provides insights into different aspects of the model's learning. For instance, consistent errors in column names may indicate a need for better schema grounding, while syntax errors may point to a need for more focused training on SQL syntax.


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
