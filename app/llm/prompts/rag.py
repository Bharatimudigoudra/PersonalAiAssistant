"""
RAG prompt templates.
"""

RAG_PROMPT = """
Use ONLY the following context.

Context:

{context}

Question:

{question}

If the answer is not found in the context,
say that the information is unavailable.
""".strip()