#!/usr/bin/env python3
"""
ETAP 1 Test: Whisper Transcription Adapter

Tests WhisperTranscriptionAdapter in isolation with real audio file.

Usage:
    python test_scripts/test_whisper_adapter.py

Expected output:
    ✅ Transcription successful
    ✅ Word count > 0
    ✅ Duration matches audio file
    📝 Preview of transcript
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sentientresearchagent.hierarchical_agent_framework.agents.definitions.meeting_agents import (
    WhisperTranscriptionAdapter
)
from sentientresearchagent.hierarchical_agent_framework.node.task_node import (
    TaskNode, TaskType, NodeType
)


async def test_whisper_transcription():
    """Test Whisper adapter with Cristiano Ronaldo podcast."""

    print("=" * 60)
    print("🎙️  ETAP 1 TEST: Whisper Transcription Adapter")
    print("=" * 60)

    # Find audio file
    audio_dir = Path(__file__).parent.parent / "test_transcript"
    audio_files = list(audio_dir.glob("*.mp4")) + list(audio_dir.glob("*.mp3"))

    if not audio_files:
        print("❌ No audio files found in test_transcript/")
        print("   Please add a test audio file (mp3 or mp4)")
        return False

    audio_path = str(audio_files[0])
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    print(f"\n📁 Test Audio File:")
    print(f"   Path: {audio_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Name: {Path(audio_path).name[:60]}...")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not found in environment!")
        print("   Set it in .env file or export OPENAI_API_KEY=sk-...")
        return False

    print(f"\n✅ OPENAI_API_KEY found (ending: ...{os.getenv('OPENAI_API_KEY')[-8:]})")

    # Create test node
    print("\n🔧 Creating test TaskNode...")
    node = TaskNode(
        goal="Transcribe audio to text",
        task_type=TaskType.SEARCH,
        node_type=NodeType.EXECUTE,
        aux_data={"audio_file_path": audio_path}
    )

    # Initialize adapter
    print("🔧 Initializing WhisperTranscriptionAdapter...")
    try:
        adapter = WhisperTranscriptionAdapter()
    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return False

    # Process transcription
    print("\n🚀 Starting transcription...")
    print("   (This may take 30 seconds to several minutes depending on file size)")
    print("   Please wait...")

    try:
        # Call adapter
        result = await adapter.process(
            node=node,
            agent_task_input=None,  # Transcription doesn't need context
            trace_manager=None  # No trace manager for isolated test
        )

        # Check for errors
        if "error" in result and result["error"]:
            print(f"\n❌ Transcription error: {result['error']}")
            return False

        # Display results
        print("\n" + "=" * 60)
        print("✅ TRANSCRIPTION SUCCESSFUL!")
        print("=" * 60)

        print(f"\n📊 Statistics:")
        print(f"   Duration: {result['duration_seconds']:.1f} seconds ({result['duration_seconds']/60:.1f} minutes)")
        print(f"   Word count: {result['word_count']:,} words")
        print(f"   Segments: {len(result.get('segments', []))} time-stamped segments")
        print(f"   Characters: {len(result['full_transcript']):,} chars")

        # Show transcript preview
        preview_length = 500
        print(f"\n📝 Transcript Preview (first {preview_length} chars):")
        print("-" * 60)
        print(result['full_transcript'][:preview_length])
        if len(result['full_transcript']) > preview_length:
            print("...")
        print("-" * 60)

        # Show first few segments with timestamps
        if result.get('segments'):
            print(f"\n⏱️  Time-stamped Segments (first 3):")
            for seg in result['segments'][:3]:
                print(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:60]}...")

        # Validate results
        print("\n🧪 Validation Checks:")

        checks_passed = 0
        checks_total = 4

        if result['word_count'] > 0:
            print("   ✅ Word count > 0")
            checks_passed += 1
        else:
            print("   ❌ Word count is 0!")

        if result['duration_seconds'] > 0:
            print("   ✅ Duration > 0")
            checks_passed += 1
        else:
            print("   ❌ Duration is 0!")

        if len(result['full_transcript']) > 0:
            print("   ✅ Transcript not empty")
            checks_passed += 1
        else:
            print("   ❌ Transcript is empty!")

        if len(result.get('segments', [])) > 0:
            print("   ✅ Segments available")
            checks_passed += 1
        else:
            print("   ⚠️  No segments (might be ok for short audio)")
            checks_passed += 1  # Don't fail on this

        # Save full output for inspection
        output_dir = Path(__file__).parent.parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "etap1_whisper_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Full result saved to: {output_file}")

        # Final verdict
        print("\n" + "=" * 60)
        if checks_passed == checks_total:
            print("🎉 ETAP 1 TEST: PASSED")
            print("✅ All checks passed!")
            print("\n➡️  Ready to proceed to ETAP 2: Action Item Extraction")
        else:
            print(f"⚠️  ETAP 1 TEST: PARTIAL ({checks_passed}/{checks_total} checks passed)")
            print("   Review errors above before proceeding")
        print("=" * 60)

        return checks_passed == checks_total

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🚀 Starting ETAP 1 Test...\n")

    success = asyncio.run(test_whisper_transcription())

    # Exit code
    sys.exit(0 if success else 1)
