from app.memory.manager import MemoryManager


def test_memory_manager():

    memory = MemoryManager()

    memory.add_user_message("Hello")

    memory.add_assistant_message("Hi Bharati!")

    history = memory.get_history()

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"

    memory.clear()

    assert memory.size() == 0