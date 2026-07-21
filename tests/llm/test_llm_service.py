from app.llm.services.llm_service import LLMService


def test_generate():

    llm = LLMService()

    response = llm.generate(
        "What is Python?"
    )

    print("\n")
    print("=" * 80)
    print(response)
    print("=" * 80)

    assert isinstance(response, str)
    assert len(response) > 0