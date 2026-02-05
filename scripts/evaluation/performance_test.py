#!/usr/bin/env python3
"""Performance testing for agent.

Usage:
    python scripts/evaluation/performance_test.py
    python scripts/evaluation/performance_test.py --duration 60 --concurrency 10
    python scripts/evaluation/performance_test.py --requests 100
"""

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

from agent_engine.evaluation import PerformanceMetrics, TestCaseLoader

# Sample queries for performance testing
SAMPLE_QUERIES = [
    "What is the capital of France?",
    "Calculate 123 * 456",
    "What time is it?",
    "Summarize the concept of machine learning in one sentence.",
    "What is 15% of 200?",
    "Explain the water cycle briefly.",
    "What is the square root of 144?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "Convert 100 Celsius to Fahrenheit.",
]


async def mock_agent_query_async(query: str) -> str:
    """Mock async agent query with simulated latency."""
    # Simulate variable latency (100ms - 2s)
    latency = random.uniform(0.1, 2.0)
    await asyncio.sleep(latency)

    # Simulate occasional errors (5% error rate)
    if random.random() < 0.05:
        raise Exception("Simulated error")

    return f"Response to: {query[:50]}..."


async def run_load_test(
    queries: list[str],
    duration_seconds: float | None = None,
    max_requests: int | None = None,
    concurrency: int = 10,
) -> PerformanceMetrics:
    """Run load test against the agent.

    Args:
        queries: List of queries to use
        duration_seconds: Test duration (None for request-based)
        max_requests: Maximum requests (None for duration-based)
        concurrency: Number of concurrent requests

    Returns:
        PerformanceMetrics with results
    """
    metrics = PerformanceMetrics()
    semaphore = asyncio.Semaphore(concurrency)

    async def make_request(query: str) -> None:
        async with semaphore:
            start_time = time.perf_counter()
            try:
                await mock_agent_query_async(query)
                latency_ms = (time.perf_counter() - start_time) * 1000
                metrics.add_latency(latency_ms)
            except TimeoutError:
                metrics.add_timeout()
            except Exception:
                metrics.add_error()

    if duration_seconds:
        # Duration-based test
        end_time = time.time() + duration_seconds
        tasks = []

        while time.time() < end_time:
            query = random.choice(queries)
            tasks.append(asyncio.create_task(make_request(query)))

            # Control request rate
            await asyncio.sleep(0.01)

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    elif max_requests:
        # Request-based test
        tasks = []
        for i in range(max_requests):
            query = queries[i % len(queries)]
            tasks.append(asyncio.create_task(make_request(query)))

        await asyncio.gather(*tasks, return_exceptions=True)

    return metrics


def print_results(metrics: PerformanceMetrics, duration: float) -> None:
    """Print performance test results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 60)

    print("\nRequests:")
    print(f"  Total: {metrics.request_count}")
    print(f"  Successful: {metrics.request_count - metrics.error_count - metrics.timeout_count}")
    print(f"  Errors: {metrics.error_count}")
    print(f"  Timeouts: {metrics.timeout_count}")
    print(f"  Error Rate: {metrics.error_rate:.2%}")

    print("\nLatency (ms):")
    print(f"  Min: {metrics.min_ms:.0f}")
    print(f"  Mean: {metrics.mean_ms:.0f}")
    print(f"  P50: {metrics.p50_ms:.0f}")
    print(f"  P90: {metrics.p90_ms:.0f}")
    print(f"  P99: {metrics.p99_ms:.0f}")
    print(f"  Max: {metrics.max_ms:.0f}")

    if duration > 0:
        throughput = metrics.request_count / duration
        print("\nThroughput:")
        print(f"  {throughput:.1f} requests/second")

    # SLA check
    meets_sla, violations = metrics.meets_sla(
        p50_threshold_ms=2000.0,
        p99_threshold_ms=10000.0,
        error_rate_threshold=0.05,
    )

    print("\nSLA Compliance:")
    if meets_sla:
        print("  All requirements met")
    else:
        print("  Violations:")
        for v in violations:
            print(f"    - {v}")

    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run performance test")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--requests",
        "-r",
        type=int,
        default=100,
        help="Number of requests (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--test-data",
        help="Path to test data JSON file (uses sample queries if not provided)",
    )
    parser.add_argument(
        "--p50-threshold",
        type=float,
        default=2000.0,
        help="P50 latency threshold in ms (default: 2000)",
    )
    parser.add_argument(
        "--p99-threshold",
        type=float,
        default=10000.0,
        help="P99 latency threshold in ms (default: 10000)",
    )

    args = parser.parse_args()

    # Load queries
    if args.test_data:
        test_data_path = Path(args.test_data)
        test_cases = TestCaseLoader.load_from_json(test_data_path)
        queries = [tc.input for tc in test_cases]
        print(f"Loaded {len(queries)} queries from {test_data_path}")
    else:
        queries = SAMPLE_QUERIES
        print(f"Using {len(queries)} sample queries")

    print("\nConfiguration:")
    print(f"  Duration: {args.duration}s" if args.duration else f"  Requests: {args.requests}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  P50 Threshold: {args.p50_threshold}ms")
    print(f"  P99 Threshold: {args.p99_threshold}ms")

    print("\nRunning performance test...")
    start_time = time.time()

    # Run test
    metrics = asyncio.run(
        run_load_test(
            queries=queries,
            duration_seconds=args.duration,
            max_requests=None if args.duration else args.requests,
            concurrency=args.concurrency,
        )
    )

    duration = time.time() - start_time

    # Print results
    print_results(metrics, duration)

    # Check SLA
    meets_sla, _ = metrics.meets_sla(
        p50_threshold_ms=args.p50_threshold,
        p99_threshold_ms=args.p99_threshold,
    )

    return 0 if meets_sla else 1


if __name__ == "__main__":
    sys.exit(main())
