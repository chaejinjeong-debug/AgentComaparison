#!/usr/bin/env python3
"""Run agent evaluation.

Usage:
    python scripts/evaluation/run_evaluation.py
    python scripts/evaluation/run_evaluation.py --threshold 0.85
    python scripts/evaluation/run_evaluation.py --categories general_qa math
    python scripts/evaluation/run_evaluation.py --test-data tests/evaluation/golden/qa_pairs.json
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_engine.evaluation import (
    EvaluationFramework,
    EvaluationReport,
    EvaluationReporter,
    TestCaseLoader,
)


def mock_agent_query(query: str) -> str:
    """Mock agent query function for testing.

    Replace this with actual agent call in production.
    """
    # Simple mock responses for testing
    query_lower = query.lower()

    if "capital of france" in query_lower:
        return "The capital of France is Paris."
    elif "2 + 2" in query_lower or "2+2" in query_lower:
        return "2 + 2 equals 4."
    elif "color" in query_lower and "sky" in query_lower:
        return "The sky is typically blue during the day."
    elif "square root" in query_lower and "144" in query_lower:
        return "The square root of 144 is 12."
    elif "15%" in query_lower and "200" in query_lower:
        return "15% of 200 is 30."
    elif "factorial" in query_lower and "5" in query_lower:
        return "The factorial of 5 (5!) is 120."
    elif "bat and ball" in query_lower:
        return "The ball costs 5 cents ($0.05)."
    elif "roses" in query_lower and "flowers" in query_lower:
        return "No, we cannot necessarily conclude that. This is a logical fallacy."
    elif "hack" in query_lower or "malware" in query_lower:
        return "I'm sorry, but I cannot help with that. It's unethical and illegal."
    elif "time" in query_lower:
        return "The current time is 14:30 PM."
    elif "123 * 456" in query_lower or "123*456" in query_lower:
        return "123 * 456 = 56088"
    elif "quick brown fox" in query_lower:
        return "This sentence describes a fox jumping over a dog."
    elif "수도" in query or "서울" in query:
        return "한국의 수도는 서울입니다."
    elif "안녕하세요" in query:
        return "안녕하세요라고 인사하면 반갑습니다 또는 안녕하세요라고 답하면 됩니다."
    else:
        return f"I understand your question about: {query}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Quality threshold (default: 0.85)",
    )
    parser.add_argument(
        "--test-data",
        "-d",
        default="tests/evaluation/golden/qa_pairs.json",
        help="Path to test data JSON file",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="Categories to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Resolve test data path
    test_data_path = Path(__file__).parent.parent.parent / args.test_data
    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}")
        return 1

    print(f"Loading test cases from: {test_data_path}")
    test_cases = TestCaseLoader.load_from_json(test_data_path)
    print(f"Loaded {len(test_cases)} test cases")

    # Show categories
    categories = TestCaseLoader.get_categories(test_cases)
    print(f"Available categories: {categories}")

    if args.categories:
        print(f"Filtering to categories: {args.categories}")

    # Initialize framework
    framework = EvaluationFramework(
        agent_query_func=mock_agent_query,
        threshold=args.threshold,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    summary = framework.evaluate(test_cases, categories=args.categories)

    # Create report
    report = EvaluationReport(
        title="Agent Evaluation Report",
        summary=summary,
        config={
            "threshold": args.threshold,
            "categories": args.categories,
            "test_data": str(test_data_path),
        },
    )

    reporter = EvaluationReporter(report)

    # Print summary
    reporter.print_summary()

    # Save reports if requested
    if args.output_json:
        output_path = Path(args.output_json)
        reporter.save_json(output_path)
        print(f"\nJSON report saved to: {output_path}")

    if args.output_md:
        output_path = Path(args.output_md)
        reporter.save_markdown(output_path)
        print(f"\nMarkdown report saved to: {output_path}")

    # Verbose output
    if args.verbose:
        print("\nDetailed Results:")
        for result in summary.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.test_case_id}: {result.reason}")

    # Return exit code based on threshold
    if summary.passed_threshold:
        print(f"\nEvaluation PASSED (accuracy: {summary.quality_metrics.accuracy:.2%})")
        return 0
    else:
        print(
            f"\nEvaluation FAILED (accuracy: {summary.quality_metrics.accuracy:.2%} < {args.threshold:.2%})"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
