"""Simple logging utility for structured data persistence."""

import datetime
import json
import os
from dataclasses import dataclass


class Logger:
    """Logger class for appending structured log entries to a JSON file with timestamps."""

    def __init__(self, log_path: str):
        """Initialize the Logger with a specified log file path.

        Args:
            log_path (str): The file path where logs will be stored (e.g., "./logs/rag_agent.json").
        """
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, data: dict):
        """Append a dict entry to the JSON log file with a timestamp.

        Args:
            data (dict): The data to log.
        """
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data,
        }

        # Read existing logs
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                existing_logs = json.load(f)
        else:
            existing_logs = []

        # Append and write
        existing_logs.append(log_entry)
        with open(self.log_path, "w") as f:
            json.dump(existing_logs, f, indent=4)
