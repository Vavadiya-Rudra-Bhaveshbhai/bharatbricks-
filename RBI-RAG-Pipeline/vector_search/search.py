from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
index = client.get_index(
    endpoint_name="rbi_circular_vs_endpoint",
    index_name="workspace.default.gold_rbi_circular_chunks_index"
)

def retrieve_rbi_context(question, k=3):
    res = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "document", "chunks_text", "issued_on", "regulation_area", "applicable_to"],
        num_results=k
    )
    rows = res["result"]["data_array"]

    contexts = []
    for r in rows:
        contexts.append({
            "chunk_id": r[0],
            "document": r[1],
            "chunks_text": r[2],
            "issued_on": r[3],
            "regulation_area": r[4],
            "applicable_to": r[5],
            "score": r[6],
        })
    return contexts

def build_context_text(contexts):
    return "\n\n".join([
        f"""[Chunk ID: {c['chunk_id']}]
Document: {c['document']}
Issued on: {c['issued_on']}
Regulation area: {c['regulation_area']}
Applicable to: {c['applicable_to']}
Text: {c['chunks_text']}"""
        for c in contexts
    ])

question = "What did RBI say about KYC updates?"
contexts = retrieve_rbi_context(question, k=3)
context_text = build_context_text(contexts)

print(context_text[:5000])


from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
index = client.get_index(
    endpoint_name="rbi_circular_vs_endpoint",
    index_name="workspace.default.gold_rbi_circular_chunks_index"
)

def retrieve_rbi_context(question, k=3):
    res = index.similarity_search(
        query_text=question,
        columns=["chunk_id", "document", "chunks_text", "issued_on", "regulation_area", "applicable_to"],
        num_results=k
    )
    rows = res["result"]["data_array"]

    contexts = []
    for r in rows:
        contexts.append({
            "chunk_id": r[0],
            "document": r[1],
            "chunks_text": r[2],
            "issued_on": r[3],
            "regulation_area": r[4],
            "applicable_to": r[5],
            "score": r[6],
        })
    return contexts

def build_context_text(contexts):
    return "\n\n".join([
        f"""[Chunk ID: {c['chunk_id']}]
Document: {c['document']}
Issued on: {c['issued_on']}
Regulation area: {c['regulation_area']}
Applicable to: {c['applicable_to']}
Text: {c['chunks_text']}"""
        for c in contexts
    ])

def build_prompt(question, context_text):
    return f"""
You are an RBI circular explainer for ordinary Indian users.

Answer only from the provided RBI context.
If the answer is not clearly present in the context, say that clearly.

Question:
{question}

RBI Context:
{context_text}

Return the answer in this format:

1. Simple explanation
2. Who it applies to
3. Key action points
4. Source chunk IDs used
"""


question = "What did RBI say about KYC updates?"
contexts = retrieve_rbi_context(question, k=3)
context_text = build_context_text(contexts)
prompt = build_prompt(question, context_text)

print(prompt[:6000])
