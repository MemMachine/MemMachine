"""Composition of the text before content: the timestamp, then the context parts, per DateTimeFormat."""

import datetime
from collections.abc import Iterable

from babel.dates import format_date, format_time, get_datetime_format

from .data_types import Context, DateTimeFormat, DateTimeStyle

# CLDR datetime style levels, ordered from compact to verbose.
_DATETIME_STYLE_LEVELS: tuple[DateTimeStyle, ...] = ("short", "medium", "long", "full")


def format_timestamp(
    timestamp: datetime.datetime,
    datetime_format: DateTimeFormat,
) -> str:
    """
    Write a timestamp per the given format.

    Returns the empty string when both the date and time styles are None.
    """
    date_style = datetime_format.date_style
    time_style = datetime_format.time_style
    locale = datetime_format.locale
    timezone = datetime_format.timezone

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


def format_header(
    timestamp: datetime.datetime,
    context: Context,
    datetime_format: DateTimeFormat,
    parts: Iterable[str],
) -> str:
    """The text before content: the timestamp, then each listed part's contribution.

    `parts` names the context part kinds to compose, in that order; a
    part not listed contributes nothing, and a listed part that renders
    None contributes nothing.
    """
    formatted_timestamp = format_timestamp(timestamp, datetime_format)
    header = f"[{formatted_timestamp}] " if formatted_timestamp else ""
    for kind in parts:
        part = context.get(kind)
        if part is None:
            continue
        contribution = part.render(datetime_format)
        if contribution:
            header += f"{contribution}: "
    return header
