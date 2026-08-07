from pprint import pprint

from app.rag.retrieval import DocumentRetriever

retriever = DocumentRetriever()

docs = retriever.retrieve(
    "Tell me about yourself"
)

print("\nRetrieved:", len(docs))

for i, doc in enumerate(docs, 1):
    print("\n", "=" * 80)
    print(i)
    print("Distance:", doc.distance)
    pprint(doc.metadata)
    print(doc.content[:300])