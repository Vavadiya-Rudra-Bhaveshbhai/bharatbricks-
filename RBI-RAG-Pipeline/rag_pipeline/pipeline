# Complete 3-Step Pipeline Test

question = "What did RBI say about KYC updates?"

print("\n" + "="*80)
print("STEP 1: Retrieving relevant RBI circular chunks...")
print("="*80)
contexts = retrieve_rbi_context(question, k=3)
context_text = build_context_text(contexts)
print(f"Retrieved {len(contexts)} chunks")
for ctx in contexts:
    print(f"  - {ctx['document'][:50]}... (score: {ctx['score']:.4f})")

print("\n" + "="*80)
print("STEP 2: Generating answer in Hindi using Databricks Llama...")
print("="*80)
messages = [
    {
        "role": "system",
        "content": "You are an RBI circular explainer for ordinary Indian users. Answer only from the provided RBI context. If the answer is not present in the context, say so clearly. Write in simple Hindi (Devanagari script)."
    },
    {
        "role": "user",
        "content": f"""Question:
{question}

RBI Context:
{context_text}

Return the answer in Hindi in this format:
1. सरल विवरण (Simple explanation)
2. यह किन पर लागू होता है (Who it applies to)
3. मुख्य कार्रवाई बिन्दु (Key action points)
4. स्रोत चंक आईडी उपयोग किए गए (Source chunk IDs used)"""
    }
]

hindi_response = llm_client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",
    messages=messages,
    max_tokens=800
)

hindi_answer = hindi_response.choices[0].message.content
print("\nHINDI ANSWER:")
print(hindi_answer)

print("\n" + "="*80)
print("STEP 3: Translating Hindi to Tamil using Databricks Llama...")
print("="*80)
tamil_answer = translate_hindi_to_tamil_llama(hindi_answer)
print("\nTAMIL ANSWER:")
print(tamil_answer)

print("\n" + "="*80)
print("✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY")
print("="*80)
