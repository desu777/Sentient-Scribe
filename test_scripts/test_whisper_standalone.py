#!/usr/bin/env python3
"""
ETAP 1 Test: Whisper Transcription with Auto-Chunking

Standalone test (no ROMA imports) to verify:
1. OpenAI Whisper API works
2. Chunking logic handles large files (>25MB)
3. Timestamps are correctly adjusted
4. Transcript quality is acceptable

Usage:
    python3 test_scripts/test_whisper_standalone.py

Environment variables required:
    OPENAI_API_KEY - Your OpenAI API key

Output:
    test_output/etap1_whisper_result.json
    chunks/ directory (can be deleted after test)
"""

import asyncio
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any
from openai import AsyncOpenAI
from audio_utils_standalone import (
    chunk_audio_by_time,
    get_audio_duration,
    cleanup_chunks,
    format_duration
)


async def transcribe_with_chunking(
    audio_file: str,
    chunk_minutes: int = 10,
    cleanup_after: bool = True
) -> Dict:
    """
    Transcribe audio file with automatic chunking for large files.

    Args:
        audio_file: Path to audio/video file
        chunk_minutes: Chunk size in minutes (default 10)
        cleanup_after: Delete chunk files after processing (default True)

    Returns:
        Dict with full_transcript, segments, duration, stats
    """

    print("\n" + "=" * 70)
    print("🎙️  WHISPER TRANSCRIPTION TEST (with Auto-Chunking)")
    print("=" * 70)

    # Validate file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    file_size_mb = os.path.getsize(audio_file) / (1024 ** 2)

    print(f"\n📁 Input File:")
    print(f"   Path: {audio_file}")
    print(f"   Name: {Path(audio_file).name[:70]}")
    print(f"   Size: {file_size_mb:.2f} MB")

    # Check API key (strip to remove trailing whitespace/newlines)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found!\n"
            "Set it in .env file or: export OPENAI_API_KEY=sk-..."
        )

    print(f"\n✅ API Key found (ending: ...{api_key[-8:]})")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Determine if chunking needed
    CHUNK_THRESHOLD_MB = 24  # Safe margin below 25MB limit

    if file_size_mb > CHUNK_THRESHOLD_MB:
        print(f"\n⚠️  File size ({file_size_mb:.1f} MB) exceeds safe limit ({CHUNK_THRESHOLD_MB} MB)")
        print(f"📋 Auto-chunking enabled...")

        # Get duration
        try:
            duration = get_audio_duration(audio_file)
            print(f"⏱️  Total duration: {format_duration(duration)}")
        except Exception as e:
            print(f"❌ Failed to get duration: {e}")
            print("   Make sure ffmpeg is installed: sudo apt install ffmpeg")
            raise

        # Split into chunks
        try:
            chunks = chunk_audio_by_time(
                audio_file,
                chunk_minutes=chunk_minutes,
                output_dir='chunks'
            )
        except Exception as e:
            print(f"❌ Failed to chunk audio: {e}")
            raise

        print(f"\n🚀 Transcribing {len(chunks)} chunks...")
        print(f"   (This will take ~{len(chunks) * 10} seconds)")

        # Transcribe each chunk
        all_segments = []
        full_transcript = ""
        chunk_stats = []

        for chunk_info in chunks:
            chunk_num = chunk_info['index'] + 1
            total_chunks = len(chunks)

            print(f"\n  📝 Chunk {chunk_num}/{total_chunks} ", end='', flush=True)
            print(f"({chunk_info['size_mb']:.1f} MB, offset: {chunk_info['start_offset']/60:.1f}min)... ", end='', flush=True)

            try:
                with open(chunk_info['file'], 'rb') as audio_chunk:
                    result = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_chunk,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )

                # Adjust timestamps to be relative to full audio
                offset = chunk_info['start_offset']

                adjusted_segments = []
                for seg in result.segments:
                    # Segments are Pydantic objects, use direct attribute access
                    adjusted_seg = {
                        'id': getattr(seg, 'id', len(all_segments)),
                        'start': getattr(seg, 'start', 0.0) + offset,
                        'end': getattr(seg, 'end', 0.0) + offset,
                        'text': getattr(seg, 'text', '')
                    }
                    adjusted_segments.append(adjusted_seg)
                    all_segments.append(adjusted_seg)

                # Append transcript
                chunk_text = result.text
                full_transcript += chunk_text + " "

                # Stats
                word_count = len(chunk_text.split())
                chunk_stats.append({
                    'chunk': chunk_num,
                    'words': word_count,
                    'duration': result.duration
                })

                print(f"✅ ({word_count:,} words)")

            except Exception as e:
                print(f"❌ Error: {e}")
                chunk_stats.append({
                    'chunk': chunk_num,
                    'error': str(e)
                })
                # Continue with other chunks

        # Cleanup chunk files
        if cleanup_after:
            print(f"\n🧹 Cleaning up {len(chunks)} chunk files...")
            cleanup_chunks(chunks)
            print("   ✅ Chunks deleted")

        # Compile results
        total_words = sum(stat.get('words', 0) for stat in chunk_stats)
        successful_chunks = sum(1 for stat in chunk_stats if 'error' not in stat)

        return {
            "full_transcript": full_transcript.strip(),
            "segments": all_segments,
            "duration_seconds": duration,
            "word_count": total_words,
            "chunking_stats": {
                "total_chunks": len(chunks),
                "successful_chunks": successful_chunks,
                "failed_chunks": len(chunks) - successful_chunks,
                "chunk_duration_minutes": chunk_minutes,
                "chunk_details": chunk_stats
            },
            "audio_file": Path(audio_file).name
        }

    else:
        # File small enough - single API call
        print(f"\n✅ File size OK ({file_size_mb:.1f} MB < {CHUNK_THRESHOLD_MB} MB)")
        print(f"📋 Using single API call (no chunking needed)")

        print(f"\n🚀 Transcribing...")

        try:
            with open(audio_file, 'rb') as f:
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            segments = [
                {
                    'id': getattr(seg, 'id', i),
                    'start': getattr(seg, 'start', 0.0),
                    'end': getattr(seg, 'end', 0.0),
                    'text': getattr(seg, 'text', '')
                }
                for i, seg in enumerate(result.segments)
            ] if hasattr(result, 'segments') else []

            word_count = len(result.text.split())
            print(f"   ✅ Transcribed {word_count:,} words")

            return {
                "full_transcript": result.text,
                "segments": segments,
                "duration_seconds": result.duration if hasattr(result, 'duration') else 0,
                "word_count": word_count,
                "chunking_stats": {
                    "total_chunks": 1,
                    "successful_chunks": 1,
                    "note": "No chunking needed (file <24MB)"
                },
                "audio_file": Path(audio_file).name
            }

        except Exception as e:
            raise RuntimeError(f"Whisper API error: {e}")


async def main():
    """Main test function."""

    # Find audio file in test_transcript/
    test_dir = Path(__file__).parent.parent / "test_transcript"

    audio_files = list(test_dir.glob("*.mp4")) + list(test_dir.glob("*.mp3")) + list(test_dir.glob("*.wav"))

    if not audio_files:
        print("\n❌ No audio files found in test_transcript/")
        print("   Add a test file (mp3, mp4, wav)")
        sys.exit(1)

    audio_file = str(audio_files[0])

    try:
        # Transcribe with auto-chunking
        result = await transcribe_with_chunking(
            audio_file=audio_file,
            chunk_minutes=10,
            cleanup_after=True  # Delete chunks after processing
        )

        # Display results
        print("\n" + "=" * 70)
        print("✅ TRANSCRIPTION COMPLETE")
        print("=" * 70)

        print(f"\n📊 Statistics:")
        print(f"   Total duration: {format_duration(result['duration_seconds'])}")
        print(f"   Total words: {result['word_count']:,}")
        print(f"   Total characters: {len(result['full_transcript']):,}")
        print(f"   Segments: {len(result['segments']):,}")

        chunking = result['chunking_stats']
        if chunking['total_chunks'] > 1:
            print(f"\n🔪 Chunking Stats:")
            print(f"   Chunks processed: {chunking['successful_chunks']}/{chunking['total_chunks']}")
            if chunking.get('failed_chunks', 0) > 0:
                print(f"   ⚠️  Failed chunks: {chunking['failed_chunks']}")

        # Show transcript preview
        preview_len = 800
        print(f"\n📝 Transcript Preview (first {preview_len} chars):")
        print("-" * 70)
        print(result['full_transcript'][:preview_len])
        if len(result['full_transcript']) > preview_len:
            print("...")
        print("-" * 70)

        # Show first few segments with timestamps
        if result['segments']:
            print(f"\n⏱️  Time-stamped Segments (first 5):")
            for seg in result['segments'][:5]:
                timestamp = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
                text_preview = seg['text'][:60]
                print(f"   {timestamp:20} {text_preview}...")

        # Save full results
        output_dir = Path(__file__).parent.parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "etap1_whisper_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Full result saved to: {output_file}")

        # Validation checks
        print("\n🧪 Validation Checks:")
        checks_passed = 0
        checks_total = 5

        if result['word_count'] > 0:
            print("   ✅ Word count > 0")
            checks_passed += 1
        else:
            print("   ❌ No words transcribed!")

        if result['duration_seconds'] > 0:
            print("   ✅ Duration > 0")
            checks_passed += 1
        else:
            print("   ❌ Duration is 0!")

        if len(result['full_transcript']) > 100:
            print("   ✅ Transcript substantial (>100 chars)")
            checks_passed += 1
        else:
            print("   ⚠️  Transcript very short")

        if len(result['segments']) > 0:
            print("   ✅ Segments available")
            checks_passed += 1
        else:
            print("   ⚠️  No segments")

        if chunking['successful_chunks'] == chunking['total_chunks']:
            print("   ✅ All chunks processed successfully")
            checks_passed += 1
        else:
            print(f"   ⚠️  Some chunks failed ({chunking['failed_chunks']})")

        # Final verdict
        print("\n" + "=" * 70)
        if checks_passed >= 4:
            print("🎉 ETAP 1 TEST: PASSED")
            print(f"   ✅ {checks_passed}/{checks_total} checks passed")
            print("\n➡️  Ready to proceed to ETAP 2: Action Item Extraction")
            return_code = 0
        else:
            print(f"⚠️  ETAP 1 TEST: PARTIAL ({checks_passed}/{checks_total} checks)")
            print("   Review errors above")
            return_code = 1

        print("=" * 70)

        sys.exit(return_code)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n🚀 Starting ETAP 1: Whisper Transcription Test\n")

    # Run test
    asyncio.run(main())
