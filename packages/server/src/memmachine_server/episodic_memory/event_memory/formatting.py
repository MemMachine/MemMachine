"""Timestamp formatting shared by EventMemory ingestion and query paths."""

import datetime

from babel.dates import format_date, format_time, get_datetime_format

from .data_types import DateTimeStyle, FormatOptions

# CLDR datetime style levels, ordered from compact to verbose.
_DATETIME_STYLE_LEVELS: tuple[DateTimeStyle, ...] = ("short", "medium", "long", "full")


def format_timestamp(
    timestamp: datetime.datetime,
    format_options: FormatOptions,
) -> str:
    """
    Format a timestamp per the given options.

    Returns the empty string when both the date and time styles are None.
    """
    date_style = format_options.date_style
    time_style = format_options.time_style
    locale = format_options.locale
    timezone = format_options.timezone

    if date_style is None and time_style is None:
        return ""

    normalized_timestamp = (
        timestamp.astimezone(timezone) if timezone is not None else timestamp
    )

    date_string = ""
    time_string = ""

    if date_style is not None:
        date_string = format_date(
            normalized_timestamp, format=date_style, locale=locale
        )
    if time_style is not None:
        time_string = format_time(
            normalized_timestamp, format=time_style, locale=locale
        )

    if not time_string:
        return date_string
    if not date_string:
        return time_string

    connector_style = _DATETIME_STYLE_LEVELS[
        max(
            _DATETIME_STYLE_LEVELS.index(date_style),
            _DATETIME_STYLE_LEVELS.index(time_style),
        )
    ]

    template = str(get_datetime_format(connector_style, locale=locale))
    return template.replace("{1}", date_string).replace("{0}", time_string)
