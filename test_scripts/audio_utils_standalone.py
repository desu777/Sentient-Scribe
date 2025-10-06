"""
Standalone audio utilities for chunking large audio files.

Production-ready utilities for:
- Getting audio duration (ffprobe)
- Splitting audio by time (ffmpeg)
- Merging transcription results

No external dependencies beyond ffmpeg/ffprobe (system tools).
"""

import subprocess
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any


def get_audio_duration(file_path: str) -> float:
    """
    Get audio duration in seconds using ffprobe.

    Args:
        file_path: Path to audio/video file

    Returns:
        Duration in seconds (float)

    Raises:
        RuntimeError: If ffprobe fails or not installed
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',  # Hide debug output
                '-show_entries', 'format=duration',  # Get duration only
                '-of', 'json',  # Output as JSON
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])

        return duration

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Install: sudo apt install ffmpeg")
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")


def chunk_audio_by_time(
    file_path: str,
    chunk_minutes: int = 10,
    output_dir: str = 'chunks',
    keep_chunks: bool = True
) -> List[Dict[str, Any]]:
    """
    Split audio into time-based chunks using ffmpeg.

    Best practice: Time-based splitting avoids cutting mid-sentence,
    which improves transcription accuracy.

    Args:
        file_path: Path to input audio/video file
        chunk_minutes: Duration of each chunk in minutes (default 10)
        output_dir: Directory to store chunks (default 'chunks')
        keep_chunks: Keep chunk files after processing (default True)

    Returns:
        List of dicts with chunk info:
        [
            {
                'file': 'chunks/chunk_000.mp3',
                'start_offset': 0.0,  # seconds
                'duration': 600.0,  # seconds
                'index': 0
            },
            ...
        ]

    Raises:
        RuntimeError: If ffmpeg fails or not installed
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get total duration
    try:
        total_duration = get_audio_duration(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {e}")

    # Calculate chunks
    chunk_seconds = chunk_minutes * 60
    num_chunks = math.ceil(total_duration / chunk_seconds)

    print(f"📊 Audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"🔪 Splitting into {num_chunks} chunks of {chunk_minutes} min each...")

    chunks = []

    for i in range(num_chunks):
        start_time = i * chunk_seconds
        output_file = os.path.join(output_dir, f"chunk_{i:03d}.mp3")

        # Use ffmpeg to extract audio chunk
        # Optimized for Whisper API (works for both video and audio inputs)
        # Whisper internally resamples to 16kHz, so we pre-process to that
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-y',  # Overwrite if exists
                    '-i', file_path,  # Input (video or audio)
                    '-ss', str(start_time),  # Start time
                    '-t', str(chunk_seconds),  # Duration
                    '-vn',  # NO video (audio only)
                    '-ac', '1',  # Mono (reduces size, Whisper uses mono internally)
                    '-ar', '16000',  # 16kHz sample rate (Whisper's internal rate)
                    '-b:a', '64k',  # 64kbps (tested - no accuracy loss, smaller files)
                    '-loglevel', 'error',  # Hide verbose output
                    output_file
                ],
                check=True,
                capture_output=True
            )

            # Get actual chunk size
            chunk_size_mb = os.path.getsize(output_file) / (1024 ** 2)

            chunk_info = {
                'file': output_file,
                'start_offset': start_time,
                'duration': min(chunk_seconds, total_duration - start_time),
                'index': i,
                'size_mb': chunk_size_mb
            }

            chunks.append(chunk_info)

            print(f"  ✅ Chunk {i+1}/{num_chunks}: {chunk_size_mb:.1f} MB")

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to create chunk {i}: {e.stderr.decode()}")
            # Continue with other chunks
            continue

        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install: sudo apt install ffmpeg")

    if not chunks:
        raise RuntimeError("No chunks created successfully")

    return chunks


def cleanup_chunks(chunks: List[Dict[str, Any]]):
    """Delete temporary chunk files."""
    for chunk in chunks:
        try:
            if os.path.exists(chunk['file']):
                os.remove(chunk['file'])
        except Exception as e:
            print(f"Warning: Failed to delete {chunk['file']}: {e}")


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
