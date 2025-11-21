"""Module for parsing filter strings into dictionaries."""

from memmachine.common.data_types import FilterablePropertyValue


def parse_filter(filter_str: str) -> dict[str, FilterablePropertyValue]:
    """
    Parse a filter string into a dictionary.

    Args:
        filter_str (str): The filter string to parse.

    Returns:
        dict: A dictionary representation of the filter.

    """
    filter_dict: dict[str, FilterablePropertyValue] = {}
    if not filter_str:
        return filter_dict

    try:
        conditions = filter_str.split("AND")
        for condition in conditions:
            if "=" in condition:
                key, value = condition.split("=", 1)
                # Store as string for now - could be enhanced to parse types
                filter_dict[key.strip()] = value.strip()
    except Exception as e:
        raise ValueError(f"Invalid filter format: {filter_str}") from e
    return filter_dict
