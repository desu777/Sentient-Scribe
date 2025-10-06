#!/usr/bin/env python3
"""
ETAP 2 Test: Extraction Agents with Gemini-2.5-Pro

Tests extraction capabilities standalone (before ROMA integration):
1. InsightsExtractor - Extract key learnings from transcript
2. SpeakerAnalyzer - Identify speakers and contributions
3. TopicExtractor - Find main topics with timestamps

Uses real Ronaldo podcast transcript from ETAP 1.

Usage:
    python3 test_scripts/test_extractors_standalone.py

Environment variables required:
    OPENROUTER_API_KEY - Your OpenRouter API key

Output:
    test_output/etap2_extractions.json
"""

import asyncio
import sys
from pathlib import Path
from extractor_utils import (
    load_transcript_from_etap1,
    call_gemini_extractor,
    validate_extraction_result,
    print_extraction_summary,
    save_extraction_results
)

# Import prompts
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sentientresearchagent.hierarchical_agent_framework.agent_configs.prompts.meeting_prompts import (
    INSIGHTS_EXTRACTION_PROMPT,
    SPEAKER_ANALYSIS_PROMPT,
    TOPIC_EXTRACTION_PROMPT
)


async def test_insights_extractor(transcript: str) -> dict:
    """Test 1: Extract key insights from interview/podcast."""

    print("\n" + "="*70)
    print("🧠 TEST 1: Insights Extraction (Gemini-2.5-Pro)")
    print("="*70)

    print(f"\n📝 Transcript length: {len(transcript)} chars ({len(transcript.split())} words)")
    print("🤖 Calling Gemini-2.5-Pro via OpenRouter...")
    print("   (This may take 10-15 seconds for long transcript)")

    # Build prompt
    prompt = INSIGHTS_EXTRACTION_PROMPT.format(transcript=transcript)

    print(f"📤 Prompt length: {len(prompt)} chars")

    # Call Gemini
    result = await call_gemini_extractor(
        prompt=prompt,
        temperature=0.2,  # Low for factual extraction
        max_tokens=4000
    )

    # Validate structure
    valid = validate_extraction_result(result, ['insights'])

    if valid:
        insights = result.get('insights', [])
        print(f"\n✅ Extraction successful!")
        print(f"   Insights found: {len(insights)}")
        print(f"   Total claimed: {result.get('total_insights', len(insights))}")

        # Show first 2 insights as preview
        print(f"\n📌 Preview (first 2 insights):")
        for i, insight in enumerate(insights[:2], 1):
            print(f"\n   {i}. [{insight.get('category', 'N/A').upper()}] {insight.get('importance', 'N/A')}")
            print(f"      {insight.get('insight', 'N/A')}")
            if insight.get('quote'):
                print(f"      Quote: \"{insight['quote'][:80]}...\"")

    else:
        print(f"\n❌ Validation failed - missing expected keys")

    return result


async def test_speaker_analyzer(transcript: str) -> dict:
    """Test 2: Analyze speakers and their contributions."""

    print("\n" + "="*70)
    print("👥 TEST 2: Speaker Analysis (Gemini-2.5-Pro)")
    print("="*70)

    # Use SPEAKER_ANALYSIS_PROMPT
    prompt = SPEAKER_ANALYSIS_PROMPT.format(
        transcript=transcript,
        attendees="Unknown"  # Let Gemini discover
    )

    print("🤖 Calling Gemini to identify speakers...")

    result = await call_gemini_extractor(
        prompt=prompt,
        temperature=0.1,  # Very low for factual identification
        max_tokens=2000
    )

    # Validate
    valid = validate_extraction_result(result, ['speakers'])

    if valid:
        speakers = result.get('speakers', [])
        print(f"\n✅ Speaker analysis complete!")
        print(f"   Speakers identified: {len(speakers)}")

        # Show speaker breakdown
        for speaker in speakers:
            print(f"\n   👤 {speaker.get('name', 'Unknown')}")
            print(f"      Role: {speaker.get('role_in_meeting', 'N/A')}")
            print(f"      Speaking time: {speaker.get('speaking_time_estimate', 'N/A')}")
            key_points = speaker.get('key_points', [])
            if key_points:
                print(f"      Key points: {len(key_points)}")
                for point in key_points[:2]:  # Show first 2
                    print(f"        - {point[:70]}...")

    else:
        print(f"\n❌ Validation failed")

    return result


async def test_topic_extractor(transcript: str, segments: list) -> dict:
    """Test 3: Extract topics with time ranges."""

    print("\n" + "="*70)
    print("📑 TEST 3: Topic Extraction with Timestamps (Gemini-2.5-Pro)")
    print("="*70)

    print(f"📊 Segments available: {len(segments)}")

    # Build prompt with segments context
    prompt = TOPIC_EXTRACTION_PROMPT.format(transcript=transcript)

    print("🤖 Calling Gemini to identify topics...")

    result = await call_gemini_extractor(
        prompt=prompt,
        temperature=0.2,
        max_tokens=3000
    )

    # Validate
    valid = validate_extraction_result(result, ['topics'])

    if valid:
        topics = result.get('topics', [])
        print(f"\n✅ Topic extraction complete!")
        print(f"   Topics found: {len(topics)}")

        # Show topics
        for i, topic in enumerate(topics, 1):
            print(f"\n   {i}. {topic.get('topic', 'N/A')}")
            print(f"      Time: {topic.get('time_spent', 'N/A')}")
            resolution = topic.get('resolution', 'N/A')
            print(f"      Status: {resolution}")
            key_points = topic.get('key_points', [])
            if key_points:
                print(f"      Key points: {', '.join(key_points[:2])}")

    else:
        print(f"\n❌ Validation failed")

    return result


async def main():
    """Main test runner for ETAP 2."""

    print("\n" + "="*70)
    print("🚀 ETAP 2: Extraction Agents Test Suite")
    print("="*70)

    # Load transcript from ETAP 1
    print("\n📂 Loading transcript from ETAP 1...")
    try:
        transcript_data = load_transcript_from_etap1()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    full_transcript = transcript_data['full_transcript']
    segments = transcript_data.get('segments', [])
    duration = transcript_data.get('duration_seconds', 0)

    print(f"✅ Transcript loaded:")
    print(f"   Words: {transcript_data.get('word_count', 0):,}")
    print(f"   Duration: {duration/60:.1f} minutes")
    print(f"   Segments: {len(segments)}")

    # Check API key
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ OPENROUTER_API_KEY not found!")
        print("   Export it: export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    print(f"\n✅ OpenRouter API key found (ending: ...{os.getenv('OPENROUTER_API_KEY')[-8:]})")

    # Run all 3 extractors
    all_results = {}

    try:
        # Test 1: Insights
        print("\n🔄 Running extraction tests...")
        insights_result = await test_insights_extractor(full_transcript)
        all_results['insights'] = insights_result

        # Test 2: Speakers
        speaker_result = await test_speaker_analyzer(full_transcript)
        all_results['speakers'] = speaker_result

        # Test 3: Topics
        topic_result = await test_topic_extractor(full_transcript, segments)
        all_results['topics'] = topic_result

        # Save combined results
        save_extraction_results(all_results)

        # Final summary
        print("\n" + "="*70)
        print("📊 ETAP 2 SUMMARY")
        print("="*70)

        insights_count = len(insights_result.get('insights', []))
        speakers_count = len(speaker_result.get('speakers', []))
        topics_count = len(topic_result.get('topics', []))

        print(f"\n✅ Insights extracted: {insights_count}")
        print(f"✅ Speakers identified: {speakers_count}")
        print(f"✅ Topics found: {topics_count}")

        # Validation checks
        print("\n🧪 Validation Checks:")
        checks_passed = 0
        checks_total = 3

        if insights_count >= 5:
            print("   ✅ Insights: Found 5+ insights")
            checks_passed += 1
        else:
            print(f"   ❌ Insights: Only {insights_count} found (expected 5+)")

        if speakers_count >= 2:
            print("   ✅ Speakers: Identified 2+ speakers")
            checks_passed += 1
        else:
            print(f"   ❌ Speakers: Only {speakers_count} found (expected 2+)")

        if topics_count >= 3:
            print("   ✅ Topics: Found 3+ topics")
            checks_passed += 1
        else:
            print(f"   ❌ Topics: Only {topics_count} found (expected 3+)")

        # Final verdict
        print("\n" + "="*70)
        if checks_passed == checks_total:
            print("🎉 ETAP 2 TEST: PASSED")
            print("   ✅ All extractors working correctly")
            print("\n➡️  Ready to proceed to ETAP 3: Full ROMA Integration")
            return_code = 0
        else:
            print(f"⚠️  ETAP 2 TEST: PARTIAL ({checks_passed}/{checks_total} checks)")
            print("   Review extraction quality above")
            return_code = 1
        print("="*70)

        sys.exit(return_code)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n🚀 Starting ETAP 2: Extraction Agents Test\n")
    asyncio.run(main())
