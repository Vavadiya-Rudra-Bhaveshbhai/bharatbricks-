from databricks.vector_search.client import VectorSearchClient

endpoint_name = "rbi_circular_vs_endpoint"
index_name = "workspace.default.gold_rbi_circular_chunks_index"

client = VectorSearchClient()

print(client.list_indexes(endpoint_name))

index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
print(index.describe())

try:
    index.wait_until_ready(wait_for_updates=True, verbose=True)
    print("Index is ready")
except Exception as e:
    print("Index is still not ready")
    print(str(e))
