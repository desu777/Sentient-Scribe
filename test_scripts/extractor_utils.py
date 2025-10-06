"""
Utilities for testing extraction agents standalone.

Production-ready helpers for:
- Loading transcripts from ETAP 1 results
- Calling Gemini-2.5-Pro via OpenRouter
- Validating extraction outputs
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from openai import AsyncOpenAI


def load_transcript_from_etap1() -> Dict[str, Any]:
    """
    Load transcript from ETAP 1 test output.

    Returns:
        Dict with full_transcript, segments, duration, etc
    """
    result_file = Path(__file__).parent.parent / "test_output" / "etap1_whisper_result.json"

    if not result_file.exists():
        raise FileNotFoundError(
            f"ETAP 1 result not found: {result_file}\n"
            "Run: python3 test_scripts/test_whisper_standalone.py first"
        )

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


async def call_gemini_extractor(
    prompt: str,
    model: str = "google/gemini-2.5-pro",
    temperature: float = 0.2,
    max_tokens: int = 4000
) -> Dict[str, Any]:
    """
    Call Gemini-2.5-Pro via OpenRouter for extraction tasks.

    Args:
        prompt: Full prompt with transcript + extraction instructions
        model: Model ID (default: google/gemini-2.5-pro)
        temperature: Lower = more factual (default 0.2)
        max_tokens: Max response length

    Returns:
        Parsed JSON response from Gemini

    Raises:
        ValueError: If OPENROUTER_API_KEY not found
        RuntimeError: If API call fails
    """

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found!\n"
            "Set it in .env file or: export OPENROUTER_API_KEY=sk-or-..."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},  # Force JSON output
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Parse JSON from response
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        return result_json

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned invalid JSON: {e}\nResponse: {result_text}")
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")


def validate_extraction_result(result: Dict, expected_keys: list) -> bool:
    """
    Validate extraction result has expected structure.

    Args:
        result: Extraction result dictionary
        expected_keys: List of required keys

    Returns:
        True if valid, False otherwise
    """

    for key in expected_keys:
        if key not in result:
            print(f"   ❌ Missing key: {key}")
            return False

    return True


def print_extraction_summary(extractor_name: str, result: Dict):
    """Pretty-print extraction results for review."""

    print(f"\n{'='*60}")
    print(f"📊 {extractor_name} Results")
    print(f"{'='*60}")

    # Pretty print with indentation
    print(json.dumps(result, indent=2, ensure_ascii=False)[:1000])

    if len(json.dumps(result)) > 1000:
        print("...")
        print(f"[Truncated - see full output in test_output/]")


def save_extraction_results(results: Dict, filename: str = "etap2_extractions.json"):
    """Save all extraction results to test_output."""

    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / filename

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 All results saved to: {output_file}")
