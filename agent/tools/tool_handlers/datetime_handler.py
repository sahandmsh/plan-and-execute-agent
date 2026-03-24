import datetime
from base.data_classes import ToolResult


def get_current_datetime() -> ToolResult:
    """Gets current date and time in the format 'YYYY-MM-DD HH:MM:SS'
    Returns:
        ToolResult: The current date and time as a string.
    """
    value = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ToolResult(text=value)


def get_days_since_epoch(target_date: str) -> ToolResult:
    """Returns the number of days since Unix epoch (January 1, 1970, 00:00:00 UTC).
    Args:
        target_date (str): The target date in the format 'YYYY-MM-DD'.

    Returns:
        ToolResult: The number of days since the Unix epoch.
    """
    target_datetime = datetime.datetime.strptime(target_date, "%Y-%m-%d")
    days = (target_datetime - datetime.datetime(1970, 1, 1)).days
    return ToolResult(text=str(days), data=days)
