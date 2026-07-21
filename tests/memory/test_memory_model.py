from app.memory.models import ChatMessage


def test_chat_message():

    message = ChatMessage(
        role="user",
        content="Hello",
    )

    assert message.role == "user"
    assert message.content == "Hello"
    assert message.timestamp is not None