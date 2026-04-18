from pyspark.sql import functions as F
import pandas as pd

# Load evaluation dataset
eval_table = "workspace.default.gold_rbi_eval"
eval_df = spark.table(eval_table)

print(f"Evaluation dataset: {eval_df.count()} questions")
print("\nSample evaluation questions:")
display(eval_df.select("question", "answer", "category", "estimated_difficulty").limit(5))

# Evaluation function
def evaluate_rag_answer(question, expected_answer, generated_answer, category):
    """
    Evaluate RAG answer quality using LLM as judge
    """
    evaluation_prompt = f"""You are an expert evaluator for a question-answering system about RBI circulars.

Evaluate the generated answer against the expected answer based on:
1. Accuracy (0-10): Does it contain correct information from the context?
2. Completeness (0-10): Does it cover all key points from the expected answer?
3. Clarity (0-10): Is it clear and easy to understand?
4. Relevance (0-10): Does it directly answer the question?

Question: {question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Return ONLY a JSON with scores and brief reasoning:
{{
  "accuracy": <score>,
  "completeness": <score>,
  "clarity": <score>,
  "relevance": <score>,
  "overall_score": <average>,
  "reasoning": "<brief explanation>"
}}"""
    
    messages = [
        {"role": "system", "content": "You are an evaluation expert. Return only valid JSON."},
        {"role": "user", "content": evaluation_prompt}
    ]
    
    response = llm_client.chat.completions.create(
        model="databricks-meta-llama-3-3-70b-instruct",
        messages=messages,
        max_tokens=500,
        temperature=0.1
    )
    
    return response.choices[0].message.content

# Run evaluation on sample questions
print("\n" + "="*80)
print("Running Evaluation on Sample Questions")
print("="*80)

eval_sample = eval_df.limit(3).toPandas()
evaluation_results = []

for idx, row in eval_sample.iterrows():
    print(f"\n[{idx+1}/{len(eval_sample)}] Evaluating: {row['question'][:60]}...")
    
    # Generate answer using RAG pipeline
    result = answer_rbi_question(
        question=row['question'],
        target_language="hindi",
        num_chunks=3
    )
    
    generated_answer = result['hindi_answer']
    
    # Evaluate
    eval_result = evaluate_rag_answer(
        question=row['question'],
        expected_answer=row['answer'],
        generated_answer=generated_answer,
        category=row['category']
    )
    
    evaluation_results.append({
        'qa_id': row['qa_id'],
        'question': row['question'],
        'category': row['category'],
        'difficulty': row['estimated_difficulty'],
        'expected_answer': row['answer'],
        'generated_answer': generated_answer,
        'evaluation': eval_result
    })
    
    print(f"Evaluation: {eval_result[:200]}...")

# Save evaluation results
eval_results_df = spark.createDataFrame(pd.DataFrame(evaluation_results))
eval_results_table = "workspace.default.rbi_rag_evaluation_results"

eval_results_df.write.format("delta").mode("overwrite").saveAsTable(eval_results_table)

print(f"\n✅ Evaluation complete! Results saved to: {eval_results_table}")
print(f"\nTo view results: spark.table('{eval_results_table}')")

# Display summary
print("\n" + "="*80)
print("Evaluation Summary")
print("="*80)
display(eval_results_df)
