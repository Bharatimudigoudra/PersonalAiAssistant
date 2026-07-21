from app.memory.service import MemoryService


def test_memory_service():

    memory = MemoryService()

    memory.add_user_message("Hello")

    memory.add_assistant_message("Hi!")

    assert memory.size() == 2

    history = memory.history()

    assert history[0].content == "Hello"

    assert history[1].content == "Hi!"

    memory.clear()

    assert memory.size() == 0