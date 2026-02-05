#!/usr/bin/env python3
"""Google Gen AI Evaluation Service PoC.

VertexAI Agent Engine의 Gen AI Evaluation 서비스를 테스트하는 PoC 스크립트.

Prerequisites:
    pip install google-cloud-aiplatform[adk,agent_engines,evaluation]>=1.112.0
    pip install pandas

Usage:
    # 기본 실행 (배포된 에이전트 필요)
    python scripts/evaluation/run_genai_evaluation.py \
        --project my-project \
        --location us-central1 \
        --agent-resource projects/my-project/locations/us-central1/agentEngines/my-agent \
        --bucket gs://my-bucket/evaluations

    # Dry-run 모드 (API 호출 없이 설정 확인)
    python scripts/evaluation/run_genai_evaluation.py \
        --project my-project \
        --agent-resource projects/my-project/locations/us-central1/agentEngines/my-agent \
        --dry-run

    # 특정 메트릭만 평가
    python scripts/evaluation/run_genai_evaluation.py \
        --project my-project \
        --agent-resource ... \
        --metrics FINAL_RESPONSE_QUALITY TOOL_USE_QUALITY

    # 커스텀 테스트 데이터 사용
    python scripts/evaluation/run_genai_evaluation.py \
        --project my-project \
        --agent-resource ... \
        --test-data my_test_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Available metrics mapping
AVAILABLE_METRICS = {
    "FINAL_RESPONSE_QUALITY": "최종 응답 품질",
    "TOOL_USE_QUALITY": "도구 사용 품질",
    "HALLUCINATION": "환각 현상 탐지",
    "SAFETY": "안전성 검사",
    "GENERAL_QUALITY": "범용 품질 평가",
    "INSTRUCTION_FOLLOWING": "지시사항 준수도",
    "TEXT_QUALITY": "텍스트 품질",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Google Gen AI Evaluation Service PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    %(prog)s --project my-project --agent-resource <resource-name> --bucket gs://bucket/path

    # Dry-run mode
    %(prog)s --project my-project --agent-resource <resource-name> --dry-run

    # Specific metrics only
    %(prog)s --project my-project --agent-resource <resource-name> --metrics FINAL_RESPONSE_QUALITY SAFETY
        """,
    )

    parser.add_argument(
        "--project",
        "-p",
        required=True,
        help="GCP Project ID",
    )
    parser.add_argument(
        "--location",
        "-l",
        default="us-central1",
        help="GCP Region (default: us-central1)",
    )
    parser.add_argument(
        "--agent-resource",
        "-a",
        required=True,
        help="Deployed Agent Engine resource name (projects/*/locations/*/agentEngines/*)",
    )
    parser.add_argument(
        "--bucket",
        "-b",
        help="GCS bucket for storing evaluation results (e.g., gs://bucket/path)",
    )
    parser.add_argument(
        "--test-data",
        "-d",
        default="tests/evaluation/golden/qa_pairs.json",
        help="Path to test data JSON file (default: tests/evaluation/golden/qa_pairs.json)",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        choices=list(AVAILABLE_METRICS.keys()),
        default=["FINAL_RESPONSE_QUALITY", "TOOL_USE_QUALITY", "HALLUCINATION", "SAFETY"],
        help="Metrics to evaluate (default: all agent metrics)",
    )
    parser.add_argument(
        "--user-id",
        default="eval_user",
        help="User ID for session context (default: eval_user)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: show configuration without making API calls",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args(argv)


def load_test_data(test_data_path: Path) -> list[dict]:
    """Load test cases from JSON file.

    Args:
        test_data_path: Path to the test data JSON file

    Returns:
        List of test case dictionaries
    """
    with open(test_data_path) as f:
        data = json.load(f)

    return data.get("test_cases", [])


def print_config(args: argparse.Namespace, test_cases: list[dict]) -> None:
    """Print configuration summary."""
    print("=" * 60)
    print("Google Gen AI Evaluation Service PoC")
    print("=" * 60)
    print(f"\nProject: {args.project}")
    print(f"Location: {args.location}")
    print(f"Agent Resource: {args.agent_resource}")
    print(f"GCS Bucket: {args.bucket or '(not specified - required for actual run)'}")
    print(f"User ID: {args.user_id}")
    print(f"\nMetrics to evaluate:")
    for metric in args.metrics:
        print(f"  - {metric}: {AVAILABLE_METRICS[metric]}")
    print(f"\nTest cases: {len(test_cases)}")

    # Show categories
    categories = {}
    for tc in test_cases:
        cat = tc.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} cases")


def run_dry_mode(args: argparse.Namespace, test_cases: list[dict]) -> int:
    """Run in dry-run mode without API calls.

    Args:
        args: Parsed arguments
        test_cases: List of test cases

    Returns:
        Exit code
    """
    print("\n" + "=" * 60)
    print("DRY-RUN MODE - No API calls will be made")
    print("=" * 60)

    print_config(args, test_cases)

    print("\n[DRY-RUN] Would create DataFrame with columns: ['prompt', 'session_inputs']")
    print(f"[DRY-RUN] Would call client.evals.run_inference(agent={args.agent_resource}, ...)")
    print(f"[DRY-RUN] Would call client.evals.create_evaluation_run(metrics={args.metrics}, ...)")
    print("[DRY-RUN] Would call client.evals.get_evaluation_run(...)")

    print("\nSample prompts that would be sent:")
    for i, tc in enumerate(test_cases[:5]):
        print(f"  {i + 1}. [{tc.get('category', 'unknown')}] {tc.get('input', '')[:60]}...")

    if len(test_cases) > 5:
        print(f"  ... and {len(test_cases) - 5} more")

    print("\nDry-run completed successfully.")
    print("To run actual evaluation, remove --dry-run flag and ensure --bucket is specified.")

    return 0


def run_evaluation(args: argparse.Namespace, test_cases: list[dict]) -> int:
    """Run the actual evaluation using Google Gen AI Evaluation Service.

    Args:
        args: Parsed arguments
        test_cases: List of test cases

    Returns:
        Exit code
    """
    # Import dependencies here to allow dry-run without them
    try:
        import pandas as pd
        import vertexai
        from google.genai import types as genai_types
        from vertexai import Client, types
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("\nPlease install with:")
        print("  pip install google-cloud-aiplatform[adk,agent_engines,evaluation]>=1.112.0")
        print("  pip install pandas")
        return 1

    if not args.bucket:
        print("Error: --bucket is required for actual evaluation run")
        print("Example: --bucket gs://my-bucket/evaluations")
        return 1

    print_config(args, test_cases)

    print("\n" + "-" * 60)
    print("Starting evaluation...")
    print("-" * 60)

    # 1. Initialize Vertex AI
    print("\n[1/5] Initializing Vertex AI client...")
    vertexai.init(project=args.project, location=args.location)

    client = Client(
        project=args.project,
        location=args.location,
        http_options=genai_types.HttpOptions(api_version="v1beta1"),
    )
    print("  Client initialized successfully")

    # 2. Prepare dataset
    print("\n[2/5] Preparing evaluation dataset...")
    session_inputs = types.evals.SessionInput(
        user_id=args.user_id,
        state={},
    )

    prompts = [tc.get("input", "") for tc in test_cases]

    agent_dataset = pd.DataFrame({
        "prompt": prompts,
        "session_inputs": [session_inputs] * len(prompts),
    })
    print(f"  Created DataFrame with {len(agent_dataset)} rows")

    # 3. Run inference
    print("\n[3/5] Running inference on agent...")
    print(f"  Agent: {args.agent_resource}")

    try:
        agent_dataset_with_inference = client.evals.run_inference(
            agent=args.agent_resource,
            src=agent_dataset,
        )
        print("  Inference completed successfully")

        if args.verbose:
            print("\n  Inference result type:", type(agent_dataset_with_inference).__name__)
            # Try to display results if possible
            if hasattr(agent_dataset_with_inference, "to_pandas"):
                df = agent_dataset_with_inference.to_pandas()
                print("  Sample responses:")
                for i, row in df.head(3).iterrows():
                    print(f"    [{i}] Prompt: {str(row.get('prompt', ''))[:50]}...")
                    if "response" in row:
                        print(f"        Response: {str(row['response'])[:100]}...")
    except Exception as e:
        print(f"  Error during inference: {e}")
        return 1

    # 4. Create evaluation run
    print("\n[4/5] Creating evaluation run...")

    # Build metrics list
    metric_objects = []
    for metric_name in args.metrics:
        metric_obj = getattr(types.RubricMetric, metric_name, None)
        if metric_obj:
            metric_objects.append(metric_obj)
        else:
            print(f"  Warning: Unknown metric '{metric_name}', skipping")

    print(f"  Metrics: {args.metrics}")
    print(f"  Destination: {args.bucket}")

    try:
        evaluation_run = client.evals.create_evaluation_run(
            dataset=agent_dataset_with_inference,
            metrics=metric_objects,
            dest=args.bucket,
        )
        print(f"  Evaluation run created: {evaluation_run.name}")
    except Exception as e:
        print(f"  Error creating evaluation run: {e}")
        return 1

    # 5. Get results
    print("\n[5/5] Retrieving evaluation results...")

    try:
        evaluation_run = client.evals.get_evaluation_run(
            name=evaluation_run.name,
            include_evaluation_items=True,
        )

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Try to show results
        if hasattr(evaluation_run, "show"):
            evaluation_run.show()

        # Extract summary if available
        results_dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "project": args.project,
            "location": args.location,
            "agent_resource": args.agent_resource,
            "metrics": args.metrics,
            "test_case_count": len(test_cases),
            "evaluation_run_name": evaluation_run.name,
        }

        # Save results if output path specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")

        print("\nEvaluation completed successfully!")
        return 0

    except Exception as e:
        print(f"  Error retrieving results: {e}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Resolve test data path
    if args.test_data.startswith("/"):
        test_data_path = Path(args.test_data)
    else:
        # Relative to project root
        project_root = Path(__file__).parent.parent.parent
        test_data_path = project_root / args.test_data

    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}")
        return 1

    # Load test cases
    try:
        test_cases = load_test_data(test_data_path)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return 1

    if not test_cases:
        print("Error: No test cases found in test data file")
        return 1

    # Run in appropriate mode
    if args.dry_run:
        return run_dry_mode(args, test_cases)
    else:
        return run_evaluation(args, test_cases)


if __name__ == "__main__":
    sys.exit(main())
