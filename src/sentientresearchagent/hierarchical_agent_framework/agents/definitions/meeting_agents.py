"""
Meeting-specific agent adapters for MeetingGenius.

This module contains specialized adapters for processing meeting recordings
and lecture videos using ROMA's hierarchical framework.

Production-ready implementation using:
- OpenAI Whisper API for transcription
- OpenRouter + google/gemini-2.5-pro for extraction/analysis
"""

import os
import json
import asyncio
import math
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from loguru import logger
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    logger.error("openai module required. Install: pip install openai")
    raise

from sentientresearchagent.hierarchical_agent_framework.agents.base_adapter import BaseAdapter
from sentientresearchagent.hierarchical_agent_framework.context.agent_io_models import AgentTaskInput
from sentientresearchagent.hierarchical_agent_framework.utils.audio_chunking import (
    chunk_audio_by_time,
    get_audio_duration,
    cleanup_chunks,
    format_duration
)

if TYPE_CHECKING:
    from sentientresearchagent.hierarchical_agent_framework.node.task_node import TaskNode
    from sentientresearchagent.hierarchical_agent_framework.tracing.manager import TraceManager


class WhisperTranscriptionAdapter(BaseAdapter):
    """
    Production-ready audio transcription using OpenAI Whisper API.

    Handles:
    - Files up to 25MB (Whisper API limit)
    - Automatic chunking for larger files
    - Time-stamped segments for lecture indexing
    - Multiple audio formats (mp3, mp4, wav, m4a)

    Returns:
    - full_transcript: Complete text transcription
    - segments: Time-stamped segments (for jumping to specific topics)
    - duration_seconds: Total audio duration
    - word_count: Number of words transcribed
    """

    adapter_name = "WhisperTranscriptionAdapter"

    # Production constants
    CHUNK_THRESHOLD_MB = 24  # Safe margin below 25MB Whisper limit
    MAX_CONCURRENT_CHUNKS = 8  # Safe concurrent limit (community tested)
    CHUNK_MINUTES = 10  # 10-minute chunks (balance between # of chunks and size)
    MAX_RETRIES = 3  # Retry failed chunks
    RETRY_DELAY_BASE = 2  # Exponential backoff base (2^attempt seconds)

    def __init__(self):
        super().__init__(self.adapter_name)

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for Whisper transcription. "
                "Get your key at: https://platform.openai.com/api-keys"
            )

        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"✅ {self.adapter_name} initialized with OpenAI Whisper API")

    async def process(
        self,
        node: "TaskNode",
        agent_task_input: AgentTaskInput,
        trace_manager: "TraceManager"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text using Whisper API with automatic parallel chunking.

        Args:
            node: TaskNode with aux_data['audio_file_path']
            agent_task_input: Context (not used for transcription)
            trace_manager: For logging execution stages

        Returns:
            Dict with transcript, segments, duration, word_count, chunking_stats
        """

        # Get audio file path from node
        audio_path = node.aux_data.get('audio_file_path')

        if not audio_path:
            # Fallback: Check environment variable (backward compatibility)
            audio_path = os.getenv('TEMP_AUDIO_PATH')

        if not audio_path or not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(f"  {self.adapter_name}: {error_msg}")
            return {
                "error": error_msg,
                "full_transcript": "",
                "duration_seconds": 0,
                "word_count": 0
            }

        logger.info(f"  {self.adapter_name}: Transcribing audio: {Path(audio_path).name}")

        # Update trace
        if trace_manager:
            trace_manager.update_stage(
                node_id=node.task_id,
                stage_name="execution",
                agent_name=self.adapter_name,
                user_input=f"Transcribe: {Path(audio_path).name}",
                model_info={"model": "whisper-1", "provider": "openai"}
            )

        try:
            # Check file size
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"  📊 File size: {file_size_mb:.2f} MB")

            # Route to appropriate transcription method
            if file_size_mb <= self.CHUNK_THRESHOLD_MB:
                logger.info(f"  ✅ File within limit - using single API call")
                result = await self._single_transcribe(audio_path, trace_manager, node.task_id)
            else:
                logger.info(f"  🔪 File exceeds {self.CHUNK_THRESHOLD_MB}MB - using parallel chunking")
                result = await self._parallel_chunked_transcribe(
                    audio_path,
                    node.task_id,
                    trace_manager
                )

            return result

        except Exception as e:
            error_msg = f"Whisper transcription error: {str(e)}"
            logger.error(f"  {self.adapter_name}: {error_msg}")

            if trace_manager:
                trace_manager.update_stage(
                    node_id=node.task_id,
                    stage_name="execution",
                    error_message=error_msg
                )

            return {
                "error": error_msg,
                "full_transcript": "",
                "duration_seconds": 0,
                "word_count": 0
            }

    async def _single_transcribe(
        self,
        audio_path: str,
        trace_manager: "TraceManager",
        task_id: str
    ) -> Dict[str, Any]:
        """Single API call for files <= 24MB."""

        with open(audio_path, 'rb') as audio_file:
            logger.info(f"  🎙️ Calling Whisper API...")

            response = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Extract data
        full_transcript = response.text
        duration = getattr(response, 'duration', 0)
        segments = getattr(response, 'segments', [])
        word_count = len(full_transcript.split())

        logger.success(f"  ✅ Transcribed {word_count:,} words from {format_duration(duration)}")

        # Update trace
        if trace_manager:
            trace_manager.update_stage(
                node_id=task_id,
                stage_name="execution",
                llm_response=full_transcript[:500] + "..." if len(full_transcript) > 500 else full_transcript,
                additional_data={
                    "duration_seconds": duration,
                    "word_count": word_count,
                    "segment_count": len(segments),
                    "chunking_method": "single"
                }
            )

        return {
            "full_transcript": full_transcript,
            "segments": [
                {
                    "id": getattr(seg, 'id', i),
                    "start": getattr(seg, 'start', 0),
                    "end": getattr(seg, 'end', 0),
                    "text": getattr(seg, 'text', '')
                }
                for i, seg in enumerate(segments)
            ] if segments else [],
            "duration_seconds": duration,
            "word_count": word_count,
            "audio_file": Path(audio_path).name,
            "chunking_stats": {
                "total_chunks": 1,
                "successful_chunks": 1,
                "method": "single"
            }
        }

    async def _parallel_chunked_transcribe(
        self,
        audio_path: str,
        task_id: str,
        trace_manager: "TraceManager"
    ) -> Dict[str, Any]:
        """
        Production-grade parallel chunking for large audio files.

        Strategy:
        1. Split audio into 10-minute chunks (ffmpeg)
        2. Process max 8 chunks concurrently (semaphore for rate limiting)
        3. Retry failed chunks (exponential backoff)
        4. Merge results maintaining timestamp order
        5. Cleanup temporary chunk files

        Args:
            audio_path: Path to large audio file (>24MB)
            task_id: Base task ID for chunk tracking
            trace_manager: For progress updates

        Returns:
            Complete transcription with merged segments
        """

        logger.info(f"  🚀 Starting parallel chunked transcription...")

        # Get duration
        try:
            duration = get_audio_duration(audio_path)
            logger.info(f"  ⏱️  Total duration: {format_duration(duration)}")
        except Exception as e:
            raise RuntimeError(f"Failed to get audio duration (ffprobe error): {e}")

        # Create chunks using ffmpeg
        try:
            chunks = chunk_audio_by_time(
                file_path=audio_path,
                chunk_minutes=self.CHUNK_MINUTES,
                output_dir='chunks',
                keep_chunks=True
            )
            logger.info(f"  🔪 Created {len(chunks)} chunks of {self.CHUNK_MINUTES}min each")
        except Exception as e:
            raise RuntimeError(f"Failed to chunk audio (ffmpeg error): {e}")

        # Parallel processing with semaphore
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHUNKS)

        async def transcribe_chunk_with_retry(chunk_info: Dict) -> Dict[str, Any]:
            """Transcribe single chunk with retry logic."""

            chunk_idx = chunk_info['index']
            chunk_id = f"{task_id}_chunk_{45300 + chunk_idx}"  # Your ID system: 45300, 45301, 45302...

            async with semaphore:
                for attempt in range(self.MAX_RETRIES):
                    try:
                        logger.info(f"  🎙️  Chunk {chunk_idx + 1}/{len(chunks)} ({chunk_id}) - attempt {attempt + 1}/{self.MAX_RETRIES}")

                        with open(chunk_info['file'], 'rb') as chunk_file:
                            response = await self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=chunk_file,
                                response_format="verbose_json",
                                timestamp_granularities=["segment"]
                            )

                        # Adjust timestamps relative to full audio
                        offset = chunk_info['start_offset']
                        adjusted_segments = []

                        for seg in response.segments:
                            adjusted_segments.append({
                                "id": getattr(seg, 'id', len(adjusted_segments)),
                                "start": getattr(seg, 'start', 0.0) + offset,
                                "end": getattr(seg, 'end', 0.0) + offset,
                                "text": getattr(seg, 'text', '')
                            })

                        word_count = len(response.text.split())
                        logger.success(f"  ✅ Chunk {chunk_idx + 1} done - {word_count:,} words")

                        return {
                            'chunk_id': chunk_id,
                            'transcript': response.text,
                            'segments': adjusted_segments,
                            'order': chunk_idx,
                            'word_count': word_count,
                            'duration': response.duration if hasattr(response, 'duration') else chunk_info['duration'],
                            'success': True
                        }

                    except Exception as e:
                        if attempt < self.MAX_RETRIES - 1:
                            # Exponential backoff
                            delay = self.RETRY_DELAY_BASE ** attempt
                            logger.warning(f"  ⚠️  Chunk {chunk_idx + 1} failed (attempt {attempt + 1}): {e}")
                            logger.info(f"  ⏳ Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                        else:
                            # Final failure
                            logger.error(f"  ❌ Chunk {chunk_idx + 1} FAILED after {self.MAX_RETRIES} attempts: {e}")
                            return {
                                'chunk_id': chunk_id,
                                'transcript': '',
                                'segments': [],
                                'order': chunk_idx,
                                'word_count': 0,
                                'success': False,
                                'error': str(e)
                            }

        # Launch ALL chunks in parallel (semaphore controls concurrency)
        logger.info(f"  ⚡ Launching {len(chunks)} chunks in parallel (max {self.MAX_CONCURRENT_CHUNKS} concurrent)...")

        chunk_tasks = [transcribe_chunk_with_retry(chunk) for chunk in chunks]
        results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Handle exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"  ❌ Chunk {i} raised exception: {result}")
                processed_results.append({
                    'chunk_id': f"{task_id}_chunk_{45300 + i}",
                    'transcript': '',
                    'segments': [],
                    'order': i,
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)

        # Sort by order (defensive - should already be ordered)
        processed_results = sorted(processed_results, key=lambda x: x['order'])

        # Merge results
        successful_chunks = [r for r in processed_results if r.get('success', False)]
        failed_chunks = [r for r in processed_results if not r.get('success', False)]

        if not successful_chunks:
            raise RuntimeError(f"All {len(chunks)} chunks failed transcription")

        full_transcript = " ".join(r['transcript'] for r in successful_chunks)
        all_segments = [seg for r in successful_chunks for seg in r['segments']]
        total_words = sum(r.get('word_count', 0) for r in successful_chunks)

        # Cleanup chunk files
        logger.info(f"  🧹 Cleaning up {len(chunks)} chunk files...")
        cleanup_chunks(chunks)

        # Final stats
        logger.success(
            f"  ✅ Parallel transcription complete: {total_words:,} words from "
            f"{len(successful_chunks)}/{len(chunks)} chunks"
        )

        if failed_chunks:
            logger.warning(f"  ⚠️  {len(failed_chunks)} chunk(s) failed - partial transcription")

        # Update trace
        if trace_manager:
            trace_manager.update_stage(
                node_id=task_id,
                stage_name="execution",
                llm_response=full_transcript[:500] + "..." if len(full_transcript) > 500 else full_transcript,
                additional_data={
                    "duration_seconds": duration,
                    "word_count": total_words,
                    "segment_count": len(all_segments),
                    "chunking_method": "parallel",
                    "total_chunks": len(chunks),
                    "successful_chunks": len(successful_chunks),
                    "failed_chunks": len(failed_chunks),
                    "chunk_duration_minutes": self.CHUNK_MINUTES
                }
            )

        return {
            "full_transcript": full_transcript,
            "segments": all_segments,
            "duration_seconds": duration,
            "word_count": total_words,
            "audio_file": Path(audio_path).name,
            "chunking_stats": {
                "total_chunks": len(chunks),
                "successful_chunks": len(successful_chunks),
                "failed_chunks": len(failed_chunks),
                "method": "parallel",
                "max_concurrent": self.MAX_CONCURRENT_CHUNKS,
                "chunk_duration_minutes": self.CHUNK_MINUTES,
                "chunk_ids": [r['chunk_id'] for r in processed_results]
            }
        }


class ActionItemExtractorAdapter(BaseAdapter):
    """
    Extracts action items from meeting transcript using Gemini-2.5-Pro.

    Returns structured list of:
    - task: What needs to be done
    - owner: Who is responsible
    - deadline: When it's due
    - priority: HIGH/MEDIUM/LOW
    - context: Why it's important
    """

    adapter_name = "ActionItemExtractorAdapter"

    def __init__(self):
        super().__init__(self.adapter_name)

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for action extraction")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.model_id = "google/gemini-2.5-pro"
        logger.info(f"✅ {self.adapter_name} initialized with {self.model_id}")

    async def process(
        self,
        node: "TaskNode",
        agent_task_input: AgentTaskInput,
        trace_manager: "TraceManager"
    ) -> Dict[str, Any]:
        """Extract action items from transcript in context."""

        # Get transcript from context (passed from transcription node)
        transcript = None

        for ctx_item in agent_task_input.relevant_context_items:
            if isinstance(ctx_item.content, dict):
                if 'full_transcript' in ctx_item.content:
                    transcript = ctx_item.content['full_transcript']
                    break

        if not transcript:
            logger.warning(f"  {self.adapter_name}: No transcript found in context")
            return {"action_items": [], "count": 0, "error": "No transcript"}

        logger.info(f"  {self.adapter_name}: Extracting action items from {len(transcript)} char transcript")

        # Load meeting-specific prompt
        from ..agent_configs.prompts.meeting_prompts import STANDUP_ACTION_PROMPT

        prompt = STANDUP_ACTION_PROMPT.format(
            transcript=transcript,
            attendees=node.aux_data.get('attendees', [])
        )

        # Update trace
        if trace_manager:
            trace_manager.update_stage(
                node_id=node.task_id,
                stage_name="execution",
                agent_name=self.adapter_name,
                user_input="Extract action items",
                model_info={"model": self.model_id, "provider": "openrouter"}
            )

        try:
            # Call Gemini via OpenRouter
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            action_data = json.loads(response.choices[0].message.content)
            action_items = action_data.get('items', [])

            logger.success(f"  {self.adapter_name}: Extracted {len(action_items)} action items")

            # Update trace
            if trace_manager:
                trace_manager.update_stage(
                    node_id=node.task_id,
                    stage_name="execution",
                    llm_response=json.dumps(action_items, indent=2),
                    additional_data={
                        "action_item_count": len(action_items),
                        "high_priority_count": sum(1 for item in action_items if item.get('priority') == 'HIGH')
                    }
                )

            return {
                "action_items": action_items,
                "count": len(action_items)
            }

        except Exception as e:
            error_msg = f"Action extraction error: {str(e)}"
            logger.error(f"  {self.adapter_name}: {error_msg}")

            if trace_manager:
                trace_manager.update_stage(
                    node_id=node.task_id,
                    stage_name="execution",
                    error_message=error_msg
                )

            return {
                "action_items": [],
                "count": 0,
                "error": error_msg
            }


# Placeholder for future extractors (Etap 2)
class StandupBlockerExtractor(BaseAdapter):
    """
    Extracts blockers from standup meetings using Gemini-2.5-Pro.

    Production adapter for ROMA integration.
    """

    adapter_name = "StandupBlockerExtractor"

    def __init__(self):
        super().__init__(self.adapter_name)
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_id = "google/gemini-2.5-pro"
        logger.info(f"✅ {self.adapter_name} initialized")

    async def process(self, node, agent_task_input, trace_manager):
        """Extract blockers from transcript in context."""

        # Get transcript from ROMA context
        transcript = self._get_transcript_from_context(agent_task_input)
        if not transcript:
            return {"blockers": [], "count": 0, "error": "No transcript"}

        # Load prompt
        from ..agent_configs.prompts.meeting_prompts import STANDUP_BLOCKER_PROMPT
        prompt = STANDUP_BLOCKER_PROMPT.format(transcript=transcript)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(response.choices[0].message.content)
            logger.success(f"  {self.adapter_name}: Found {len(result.get('blockers', []))} blockers")
            return result

        except Exception as e:
            logger.error(f"  {self.adapter_name}: {e}")
            return {"blockers": [], "count": 0, "error": str(e)}

    def _get_transcript_from_context(self, agent_task_input):
        """Helper to extract transcript from ROMA context."""
        for ctx_item in agent_task_input.relevant_context_items:
            if isinstance(ctx_item.content, dict) and 'full_transcript' in ctx_item.content:
                return ctx_item.content['full_transcript']
        return None


class StandupActionExtractor(ActionItemExtractorAdapter):
    """Alias for standup-specific action extraction."""
    adapter_name = "StandupActionExtractor"


class StandupSummaryWriter(BaseAdapter):
    """
    Writes tactical standup summary using Gemini-2.5-Pro.

    Production adapter for ROMA integration.
    """

    adapter_name = "StandupSummaryWriter"

    def __init__(self):
        super().__init__(self.adapter_name)
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_id = "google/gemini-2.5-pro"

    async def process(self, node, agent_task_input, trace_manager):
        """Write standup summary from extracted data."""

        # Get all extraction results from context
        extracted_data = {}
        for ctx_item in agent_task_input.relevant_context_items:
            if isinstance(ctx_item.content, dict):
                extracted_data.update(ctx_item.content)

        # Load prompt
        from ..agent_configs.prompts.meeting_prompts import STANDUP_SUMMARY_PROMPT
        prompt = STANDUP_SUMMARY_PROMPT.format(
            extracted_data=json.dumps(extracted_data, indent=2)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5  # Slightly higher for natural writing
            )

            summary = response.choices[0].message.content
            logger.success(f"  {self.adapter_name}: Generated summary ({len(summary)} chars)")
            return {"summary": summary}

        except Exception as e:
            logger.error(f"  {self.adapter_name}: {e}")
            return {"summary": "", "error": str(e)}


# Placeholder adapters (will implement when needed)
class DecisionExtractorAdapter(BaseAdapter):
    """Extracts key decisions - to be implemented."""
    adapter_name = "DecisionExtractorAdapter"

    def __init__(self):
        super().__init__(self.adapter_name)
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model_id = "google/gemini-2.5-pro"

    async def process(self, node, agent_task_input, trace_manager):
        logger.info(f"  {self.adapter_name}: Placeholder - returning empty")
        return {"decisions": [], "count": 0}


class SpeakerAnalyzerAdapter(BaseAdapter):
    """Analyzes speakers - to be implemented."""
    adapter_name = "SpeakerAnalyzerAdapter"

    def __init__(self):
        super().__init__(self.adapter_name)
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model_id = "google/gemini-2.5-pro"

    async def process(self, node, agent_task_input, trace_manager):
        logger.info(f"  {self.adapter_name}: Placeholder - returning empty")
        return {"speakers": [], "count": 0}
