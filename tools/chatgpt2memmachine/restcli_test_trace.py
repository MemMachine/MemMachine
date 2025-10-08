#!/usr/bin/env python3
"""
Test script to demonstrate the API tracing functionality
"""

from restcli import MemMachineRestClient


def test_tracing():
    print("Testing API tracing functionality...")

    client = MemMachineRestClient(base_url="http://localhost:8080", trace=True)

    try:
        # This will show tracing for a successful request
        print("\n=== Testing POST request tracing ===")
        client.post_episodic_memory("Test message for tracing")
    except Exception as e:
        print(f"Expected error (testing tracing): {e}")

    try:
        # This will show tracing for a search request
        print("\n=== Testing SEARCH request tracing ===")
        client.search_episodic_memory("test query", limit=3)
    except Exception as e:
        print(f"Expected error (testing tracing): {e}")


if __name__ == "__main__":
    test_tracing()
