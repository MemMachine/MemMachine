import argparse


def positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("must be a positive integer") from err

    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value
