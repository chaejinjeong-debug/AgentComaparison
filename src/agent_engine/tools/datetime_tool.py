"""DateTime tool implementation.

This module provides date and time functionality including
current time retrieval and timezone conversion.
"""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

import structlog

from agent_engine.exceptions import ToolExecutionError

logger = structlog.get_logger()

# Common timezone aliases
TIMEZONE_ALIASES: dict[str, str] = {
    "KST": "Asia/Seoul",
    "JST": "Asia/Tokyo",
    "CST": "America/Chicago",
    "EST": "America/New_York",
    "PST": "America/Los_Angeles",
    "GMT": "Europe/London",
    "UTC": "UTC",
    "CET": "Europe/Paris",
    "IST": "Asia/Kolkata",
    "AEST": "Australia/Sydney",
}


def get_current_datetime(
    timezone_name: str = "UTC",
    format_str: str = "%Y-%m-%d %H:%M:%S %Z",
) -> dict[str, Any]:
    """Get the current date and time in a specified timezone.

    Args:
        timezone_name: Timezone name (e.g., "UTC", "Asia/Seoul", "America/New_York")
                      or common alias (e.g., "KST", "EST", "PST")
        format_str: DateTime format string (default: "%Y-%m-%d %H:%M:%S %Z")

    Returns:
        Dictionary containing:
            - datetime: Formatted datetime string
            - timestamp: Unix timestamp
            - timezone: Timezone name used
            - iso_format: ISO 8601 formatted string
            - components: Dictionary with year, month, day, hour, minute, second

    Raises:
        ToolExecutionError: If timezone is invalid

    Examples:
        >>> get_current_datetime("Asia/Seoul")
        {
            "datetime": "2026-02-02 18:30:45 KST",
            "timestamp": 1738488645.123,
            "timezone": "Asia/Seoul",
            ...
        }
    """
    logger.info("get_current_datetime_executed", timezone=timezone_name)

    try:
        # Resolve timezone alias
        tz_name = _resolve_timezone(timezone_name)
        tz = ZoneInfo(tz_name)

        # Get current time in specified timezone
        now = datetime.now(tz)

        return {
            "datetime": now.strftime(format_str),
            "timestamp": now.timestamp(),
            "timezone": tz_name,
            "iso_format": now.isoformat(),
            "components": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "weekday": now.strftime("%A"),
            },
        }
    except Exception as e:
        raise ToolExecutionError(
            f"Failed to get datetime: {e}",
            tool_name="get_current_datetime",
            tool_args={"timezone_name": timezone_name},
            details={"error_type": type(e).__name__},
        ) from e


def convert_timezone(
    datetime_str: str,
    from_timezone: str,
    to_timezone: str,
    input_format: str = "%Y-%m-%d %H:%M:%S",
    output_format: str = "%Y-%m-%d %H:%M:%S %Z",
) -> dict[str, Any]:
    """Convert a datetime from one timezone to another.

    Args:
        datetime_str: DateTime string to convert
        from_timezone: Source timezone name or alias
        to_timezone: Target timezone name or alias
        input_format: Format of the input datetime string
        output_format: Format for the output datetime string

    Returns:
        Dictionary containing:
            - original: Original datetime string
            - converted: Converted datetime string
            - from_timezone: Source timezone name
            - to_timezone: Target timezone name
            - timestamp: Unix timestamp

    Raises:
        ToolExecutionError: If timezone or datetime format is invalid

    Examples:
        >>> convert_timezone("2026-02-02 09:30:00", "UTC", "Asia/Seoul")
        {
            "original": "2026-02-02 09:30:00",
            "converted": "2026-02-02 18:30:00 KST",
            "from_timezone": "UTC",
            "to_timezone": "Asia/Seoul",
            ...
        }
    """
    logger.info(
        "convert_timezone_executed",
        datetime_str=datetime_str,
        from_timezone=from_timezone,
        to_timezone=to_timezone,
    )

    try:
        # Resolve timezone aliases
        from_tz_name = _resolve_timezone(from_timezone)
        to_tz_name = _resolve_timezone(to_timezone)

        from_tz = ZoneInfo(from_tz_name)
        to_tz = ZoneInfo(to_tz_name)

        # Parse the input datetime
        dt = datetime.strptime(datetime_str, input_format)
        dt = dt.replace(tzinfo=from_tz)

        # Convert to target timezone
        converted_dt = dt.astimezone(to_tz)

        return {
            "original": datetime_str,
            "converted": converted_dt.strftime(output_format),
            "from_timezone": from_tz_name,
            "to_timezone": to_tz_name,
            "timestamp": converted_dt.timestamp(),
            "iso_format": converted_dt.isoformat(),
        }
    except ValueError as e:
        raise ToolExecutionError(
            f"Invalid datetime format: {e}",
            tool_name="convert_timezone",
            tool_args={
                "datetime_str": datetime_str,
                "from_timezone": from_timezone,
                "to_timezone": to_timezone,
            },
            details={"expected_format": input_format},
        ) from e
    except Exception as e:
        raise ToolExecutionError(
            f"Timezone conversion failed: {e}",
            tool_name="convert_timezone",
            tool_args={
                "datetime_str": datetime_str,
                "from_timezone": from_timezone,
                "to_timezone": to_timezone,
            },
            details={"error_type": type(e).__name__},
        ) from e


def _resolve_timezone(tz_input: str) -> str:
    """Resolve a timezone name or alias to a valid timezone.

    Args:
        tz_input: Timezone name or common alias

    Returns:
        Valid timezone name

    Raises:
        ToolExecutionError: If timezone is not found
    """
    # Check if it's an alias
    tz_upper = tz_input.upper()
    if tz_upper in TIMEZONE_ALIASES:
        return TIMEZONE_ALIASES[tz_upper]

    # Check if it's a valid timezone name
    if tz_input in available_timezones():
        return tz_input

    # Try case-insensitive match
    for tz in available_timezones():
        if tz.lower() == tz_input.lower():
            return tz

    raise ToolExecutionError(
        f"Unknown timezone: {tz_input}",
        tool_name="resolve_timezone",
        tool_args={"timezone": tz_input},
        details={"available_aliases": list(TIMEZONE_ALIASES.keys())},
    )


def list_available_timezones(prefix: str = "") -> dict[str, Any]:
    """List available timezones, optionally filtered by prefix.

    Args:
        prefix: Optional prefix to filter timezones (e.g., "Asia/", "America/")

    Returns:
        Dictionary containing:
            - timezones: List of matching timezone names
            - count: Number of matching timezones
            - aliases: Common timezone aliases
    """
    all_tzs = sorted(available_timezones())

    if prefix:
        filtered = [tz for tz in all_tzs if tz.lower().startswith(prefix.lower())]
    else:
        filtered = all_tzs

    return {
        "timezones": filtered,
        "count": len(filtered),
        "aliases": TIMEZONE_ALIASES,
    }
