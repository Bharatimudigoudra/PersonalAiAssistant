from app.rag.retrieval import DocumentRetriever

retriever = DocumentRetriever()

results = retriever.retrieve(
    "What is Bharati's work experience?"
)

print("\nRetrieved Chunks:\n")

for index, chunk in enumerate(results, start=1):
    print(f"Chunk {index}")
    print("-" * 60)
    print(chunk)
    print()