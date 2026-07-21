from app.rag.rag_service import RAGService

rag = RAGService()

answer = rag.ask(
    "What is Bharati's work experience?"
)

print()
print("=" * 80)
print(answer)
print("=" * 80)