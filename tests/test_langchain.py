from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.2,
)

response = llm.invoke(
    [HumanMessage(content="What is Python?")]
)

print("=" * 80)
print(type(response))
print("=" * 80)
print(response)
print("=" * 80)
print("CONTENT")
print(repr(response.content))
print("=" * 80)
print(response.response_metadata)