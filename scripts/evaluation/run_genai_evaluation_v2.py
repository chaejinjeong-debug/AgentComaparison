#!/usr/bin/env python3
"""Google Gen AI Evaluation Service PoC - V2 (Non-ADK Agent Support).

VertexAI Agent Engine의 Gen AI Evaluation 서비스를 테스트하는 PoC 스크립트.
이 버전은 ADK가 아닌 커스텀 래퍼 에이전트(PydanticAIAgentWrapper)를 지원합니다.

2단계 평가 방식:
1. 배포된 에이전트에서 응답 수집 (agent.query())
2. 수집된 응답을 평가 (client.evals.evaluate())

Prerequisites:
    pip install google-cloud-aiplatform[adk,agent_engines,evaluation]>=1.112.0
    pip install pandas

Usage:
    # 기본 실행 (배포된 에이전트 필요)
    python scripts/evaluation/run_genai_evaluation_v2.py \
        --project my-project \
        --location us-central1 \
        --agent-resource projects/my-project/locations/us-central1/agentEngines/my-agent

    # 에이전트가 asia-northeast3에 있지만 평가는 us-central1에서 실행
    python scripts/evaluation/run_genai_evaluation_v2.py \
        --project my-project \
        --location asia-northeast3 \
        --eval-location us-central1 \
        --agent-resource projects/my-project/locations/asia-northeast3/agentEngines/my-agent

    # Dry-run 모드 (API 호출 없이 설정 확인)
    python scripts/evaluation/run_genai_evaluation_v2.py \
        --project my-project \
        --agent-resource projects/my-project/locations/us-central1/agentEngines/my-agent \
        --dry-run

    # 특정 메트릭만 평가
    python scripts/evaluation/run_genai_evaluation_v2.py \
        --project my-project \
        --agent-resource ... \
        --metrics GENERAL_QUALITY TEXT_QUALITY

    # 커스텀 테스트 데이터 사용
    python scripts/evaluation/run_genai_evaluation_v2.py \
        --project my-project \
        --agent-resource ... \
        --test-data my_test_cases.json
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Available metrics for response evaluation (non-ADK)
# Note: TOOL_USE_QUALITY, HALLUCINATION require agent trace (ADK only)
AVAILABLE_METRICS = {
    "GENERAL_QUALITY": "범용 품질 평가 (권장)",
    "TEXT_QUALITY": "텍스트 품질 (유창성, 일관성, 문법)",
    "INSTRUCTION_FOLLOWING": "지시사항 준수도",
    "FLUENCY": "유창성 (1-5 점수)",
    "COHERENCE": "일관성 (1-5 점수)",
}

# Metrics that require ADK agent trace (not available in V2)
ADK_ONLY_METRICS = {
    "FINAL_RESPONSE_QUALITY",
    "TOOL_USE_QUALITY",
    "HALLUCINATION",
    "SAFETY",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Google Gen AI Evaluation Service PoC V2 (Non-ADK Agent Support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    %(prog)s --project my-project --agent-resource <resource-name>

    # Dry-run mode
    %(prog)s --project my-project --agent-resource <resource-name> --dry-run

    # Specific metrics only
    %(prog)s --project my-project --agent-resource <resource-name> --metrics GENERAL_QUALITY TEXT_QUALITY

Note: This V2 script supports non-ADK agents (like PydanticAIAgentWrapper) using a 2-step approach:
      1. Collect responses using agent.query()
      2. Evaluate responses using client.evals.evaluate()
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
        help="GCP Region where the agent is deployed (default: us-central1)",
    )
    parser.add_argument(
        "--eval-location",
        default="us-central1",
        help="GCP Region for Evaluation Service (must be us-central1, default: us-central1)",
    )
    parser.add_argument(
        "--agent-resource",
        "-a",
        required=True,
        help="Deployed Agent Engine resource name (projects/*/locations/*/agentEngines/*)",
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
        default=["GENERAL_QUALITY", "TEXT_QUALITY", "INSTRUCTION_FOLLOWING"],
        help="Metrics to evaluate (default: GENERAL_QUALITY, TEXT_QUALITY, INSTRUCTION_FOLLOWING)",
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
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Maximum number of prompts to evaluate (0 = all)",
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
    print("Google Gen AI Evaluation Service PoC V2")
    print("(Non-ADK Agent Support - 2-Step Evaluation)")
    print("=" * 60)
    print(f"\nProject: {args.project}")
    print(f"Agent Location: {args.location}")
    print(f"Eval Location: {args.eval_location}")
    print(f"Agent Resource: {args.agent_resource}")
    print(f"\nMetrics to evaluate:")
    for metric in args.metrics:
        print(f"  - {metric}: {AVAILABLE_METRICS[metric]}")

    # Warning about ADK-only metrics
    print(f"\nNote: ADK-only metrics (not available in V2):")
    for metric in sorted(ADK_ONLY_METRICS):
        print(f"  - {metric}")

    num_cases = len(test_cases)
    if args.max_prompts > 0:
        num_cases = min(num_cases, args.max_prompts)

    print(f"\nTest cases: {num_cases}")

    # Show categories
    categories: dict[str, int] = {}
    for tc in test_cases[:num_cases] if args.max_prompts > 0 else test_cases:
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

    num_cases = len(test_cases)
    if args.max_prompts > 0:
        num_cases = min(num_cases, args.max_prompts)

    print("\n[DRY-RUN] Step 1: Would collect responses from deployed agent")
    print(f"[DRY-RUN] Would call agent.query() for {num_cases} prompts")

    print("\n[DRY-RUN] Step 2: Would evaluate collected responses")
    print(f"[DRY-RUN] Would call client.evals.evaluate(metrics={args.metrics}, ...)")

    print("\nSample prompts that would be sent:")
    for i, tc in enumerate(test_cases[:5]):
        print(f"  {i + 1}. [{tc.get('category', 'unknown')}] {tc.get('input', '')[:60]}...")

    if num_cases > 5:
        print(f"  ... and {num_cases - 5} more")

    print("\nDry-run completed successfully.")
    print("To run actual evaluation, remove --dry-run flag.")

    return 0


async def collect_responses_async(
    client, agent_resource: str, prompts: list[str], verbose: bool = False
) -> list[str]:
    """Collect responses from the deployed agent (async version).

    Args:
        client: Vertex AI client
        agent_resource: Agent Engine resource name
        prompts: List of prompts to send
        verbose: Whether to print verbose output

    Returns:
        List of responses
    """
    agent = client.agent_engines.get(name=agent_resource)
    responses = []

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"  [{i + 1}/{len(prompts)}] Sending: {prompt[:50]}...")

        try:
            response = agent.query(message=prompt)

            # Handle coroutine if returned
            if inspect.iscoroutine(response):
                response = await response

            # Extract text from response
            if isinstance(response, dict):
                response_text = response.get("response", str(response))
            else:
                response_text = str(response)

            responses.append(response_text)

            if verbose:
                print(f"          Response: {response_text[:80]}...")
        except Exception as e:
            print(f"  Error querying agent for prompt {i + 1}: {e}")
            responses.append(f"[ERROR] {e}")

    return responses


def collect_responses(
    client, agent_resource: str, prompts: list[str], verbose: bool = False
) -> list[str]:
    """Collect responses from the deployed agent.

    Args:
        client: Vertex AI client
        agent_resource: Agent Engine resource name
        prompts: List of prompts to send
        verbose: Whether to print verbose output

    Returns:
        List of responses
    """
    return asyncio.run(collect_responses_async(client, agent_resource, prompts, verbose))


def run_evaluation(args: argparse.Namespace, test_cases: list[dict]) -> int:
    """Run the actual evaluation using Google Gen AI Evaluation Service.

    This V2 version uses a 2-step approach:
    1. Collect responses from deployed agent using agent.query()
    2. Evaluate responses using client.evals.evaluate()

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

    print_config(args, test_cases)

    print("\n" + "-" * 60)
    print("Starting 2-Step Evaluation...")
    print("-" * 60)

    # Apply max_prompts limit
    num_cases = len(test_cases)
    if args.max_prompts > 0:
        num_cases = min(num_cases, args.max_prompts)
        test_cases = test_cases[:num_cases]

    # 1. Initialize Vertex AI clients
    print("\n[1/4] Initializing Vertex AI clients...")

    # Agent client (for querying the agent in its deployed region)
    vertexai.init(project=args.project, location=args.location)
    agent_client = Client(
        project=args.project,
        location=args.location,
        http_options=genai_types.HttpOptions(api_version="v1beta1"),
    )
    print(f"  Agent client initialized (location: {args.location})")

    # Eval client (for evaluation service - must be in supported region like us-central1)
    eval_client = Client(
        project=args.project,
        location=args.eval_location,
        http_options=genai_types.HttpOptions(api_version="v1beta1"),
    )
    print(f"  Eval client initialized (location: {args.eval_location})")

    # 2. Collect responses from agent
    print(f"\n[2/4] Collecting responses from agent ({num_cases} prompts)...")
    print(f"  Agent: {args.agent_resource}")

    prompts = [tc.get("input", "") for tc in test_cases]

    try:
        responses = collect_responses(
            agent_client, args.agent_resource, prompts, verbose=args.verbose
        )
        print(f"  Collected {len(responses)} responses")
    except Exception as e:
        print(f"  Error collecting responses: {e}")
        return 1

    # 3. Prepare evaluation dataset
    print("\n[3/4] Preparing evaluation dataset...")

    eval_df = pd.DataFrame({
        "prompt": prompts,
        "response": responses,
    })

    print(f"  Created DataFrame with {len(eval_df)} rows")
    print("  Columns: prompt, response")

    if args.verbose:
        print("\n  Sample data:")
        for i, row in eval_df.head(3).iterrows():
            print(f"    [{i}] Prompt: {str(row['prompt'])[:50]}...")
            print(f"        Response: {str(row['response'])[:80]}...")

    # 4. Run evaluation
    print("\n[4/4] Running evaluation...")

    # Build metrics list
    metric_objects = []
    for metric_name in args.metrics:
        metric_obj = getattr(types.RubricMetric, metric_name, None)
        if metric_obj:
            metric_objects.append(metric_obj)
        else:
            print(f"  Warning: Unknown metric '{metric_name}', skipping")

    print(f"  Metrics: {args.metrics}")

    try:
        # Use evaluate() for response-only evaluation (non-ADK)
        # Note: Evaluation service must be in a supported region (us-central1)
        eval_result = eval_client.evals.evaluate(
            dataset=eval_df,
            metrics=metric_objects,
        )

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Try to show results
        if hasattr(eval_result, "show"):
            eval_result.show()

        # Extract summary
        results_dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "project": args.project,
            "location": args.location,
            "agent_resource": args.agent_resource,
            "metrics": args.metrics,
            "test_case_count": len(test_cases),
            "evaluation_method": "2-step (non-ADK)",
        }

        # Try to extract metric scores
        if hasattr(eval_result, "to_pandas"):
            result_df = eval_result.to_pandas()
            if args.verbose:
                print("\nDetailed Results:")
                print(result_df.to_string())

            # Compute summary statistics
            summary_stats = {}
            for metric in args.metrics:
                metric_col = metric.lower()
                if metric_col in result_df.columns:
                    summary_stats[metric] = {
                        "mean": float(result_df[metric_col].mean()),
                        "min": float(result_df[metric_col].min()),
                        "max": float(result_df[metric_col].max()),
                    }
            results_dict["summary_stats"] = summary_stats

            print("\nSummary Statistics:")
            for metric, stats in summary_stats.items():
                print(f"  {metric}:")
                print(f"    Mean: {stats['mean']:.3f}")
                print(f"    Min: {stats['min']:.3f}")
                print(f"    Max: {stats['max']:.3f}")

        # Save results if output path specified
        if args.output:
            output_path = Path(args.output)

            # Also save detailed results as CSV
            if hasattr(eval_result, "to_pandas"):
                csv_path = output_path.with_suffix(".csv")
                eval_result.to_pandas().to_csv(csv_path, index=False)
                print(f"\nDetailed results saved to: {csv_path}")

            with open(output_path, "w") as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            print(f"Summary results saved to: {output_path}")

        print("\nEvaluation completed successfully!")
        return 0

    except Exception as e:
        print(f"  Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
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
