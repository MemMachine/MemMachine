import argparse
import importlib.util
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="+", help="Python modules that must exist")
    return parser


def find_missing_modules(modules: list[str]) -> list[str]:
    return [module for module in modules if importlib.util.find_spec(module) is None]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing_modules = find_missing_modules(args.modules)
    if not missing_modules:
        return 0

    missing_list = ", ".join(missing_modules)
    print(
        f"Error: missing required Python module(s): {missing_list}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
