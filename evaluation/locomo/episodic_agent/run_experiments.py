# This is adapted from Mem0 (https://github.com/mem0ai/mem0/blob/main/evaluation/run_experiments.py)
# It is modified to work with MemMachine.

import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--method", default="add", help="Method to use")
    parser.add_argument(
        "--dataset",
        type=str,
        default="locomo10.json",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8080",
        help="URL of server",
    )

    args = parser.parse_args()

    print(
        f"\nMethod: {args.method}\nDataset: {args.dataset}\nBase Url: {args.base_url}"
    )

    from memmachine_locomo import MemMachineSearch

    dataset = args.dataset
    if args.method == "search":
        im = MemMachineSearch()
        asyncio.run(im.process_data_file(dataset, base_url=args.base_url))
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
