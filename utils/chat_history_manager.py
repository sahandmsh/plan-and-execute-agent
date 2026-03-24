import datetime
import os

from utils.logger import Logger


class ChatHistoryManager:
    """Manages conversation history with JSON file persistence and a sliding window.

    Each entry is a ``{"role": ..., "content": ...}`` dict.  Entries are
    appended to an in-memory list *and* immediately written to a timestamped
    JSON log file via :class:`~utils.logger.Logger`.

    When constructing the context string to pass to the LLM, only the most
    recent ``window_size`` entries are included, keeping the prompt bounded
    regardless of how long the session runs.

    Args:
        log_dir (str):     Directory where the JSON history file is written.
        window_size (int): Maximum number of history entries returned. Defaults to 100.
    """

    def __init__(self, log_dir: str = "./logs", window_size: int = 100):
        self.window_size = window_size
        self._history: list[dict] = []

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"chat_history_{timestamp}.json")
        self._logger = Logger(log_path)

    def append(self, role: str, content: str) -> None:
        """Append a single entry to the history and persist it to disk.

        Args:
            role (str):    Typically ``"user"`` or ``"agent"``.
            content (str): The message text.
        """
        entry = {"role": role, "content": content}
        self._history.append(entry)
        self._logger.log(entry)

    def get_window_history(self) -> list[dict]:
        """Return the most recent ``window_size`` entries.

        Returns:
            list[dict]: Sliding window of ``{"role", "content"}`` dicts.
        """
        return self._history[-self.window_size :]

    def get_history_str(self) -> str:
        """Return the sliding window serialised as a plain text string.

        Each line has the form ``role: content``, ready to be embedded
        directly in a prompt.

        Returns:
            str: Formatted history string, or an empty string if there is no
                 history yet.
        """
        window = self.get_window_history()
        if not window:
            return ""
        return "\n".join(f"{entry['role']}: {entry['content']}" for entry in window)

    def __len__(self) -> int:
        return len(self._history)
