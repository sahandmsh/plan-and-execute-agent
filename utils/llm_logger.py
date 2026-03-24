"""Shared LLM call logger — one timestamped JSON file per program run."""

import datetime
import os

from utils.logger import Logger


class LLMLogger:
    """Logs every LLM prompt/response pair to a single shared JSON file.

    A new file is created once at instantiation time using a timestamp, so all
    generative-model instances that receive the same ``LLMLogger`` object write
    to the same file for the lifetime of the process.

    Log entry structure:
        {
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "role":      "user" | "agent",
            "caller":    "dispatcher" | "planner" | "internal_synthesizer" | "unknown" | ...,
            "content":   <str>
        }
    """

    def __init__(self, log_dir: str = "./logs"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"llm_logs_{timestamp}.json")
        self._logger = Logger(log_path)

    def _append(self, role: str, content: str, caller: str = "unknown") -> None:
        self._logger.log({"role": role, "caller": caller, "content": content})

    def log_user(self, content: str, system_instructions=None) -> None:
        """Log the prompt sent to the model (role=user)."""
        caller = getattr(system_instructions, "__name__", "unknown")
        self._append("user", content, caller)

    def log_agent(self, content: str, system_instructions=None) -> None:
        """Log the response received from the model (role=agent)."""
        caller = getattr(system_instructions, "__name__", "unknown")
        self._append("agent", content, caller)
