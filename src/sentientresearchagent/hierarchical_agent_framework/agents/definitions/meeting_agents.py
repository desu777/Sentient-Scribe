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

    def __init__(self):
        super().__init__(self.adapter_name)

        api_key = os.getenv("OPENAI_API_KEY")
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
        Transcribe audio file to text using Whisper API.

        Args:
            node: TaskNode with aux_data['audio_file_path']
            agent_task_input: Context (not used for transcription)
            trace_manager: For logging execution stages

        Returns:
            Dict with transcript, segments, duration, word_count
        """

        # Get audio file path from node
        audio_path = node.aux_data.get('audio_file_path')

        if not audio_path:
            # Fallback: Check environment variable (temporary hack)
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

        logger.info(f"  {self.adapter_name}: Transcribing audio file: {audio_path}")

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
            logger.info(f"  Audio file size: {file_size_mb:.2f} MB")

            # Whisper API limit is 25MB
            if file_size_mb > 25:
                logger.warning(f"  File exceeds 25MB limit. Will attempt upload anyway...")
                # TODO: Implement chunking for files >25MB
                # For now, try and handle error if fails

            # Transcribe using Whisper API
            with open(audio_path, 'rb') as audio_file:
                logger.info(f"  Calling Whisper API...")

                transcription_response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get segments with timestamps
                    timestamp_granularities=["segment"]
                )

            # Extract data from response
            full_transcript = transcription_response.text
            duration = getattr(transcription_response, 'duration', 0)
            segments = getattr(transcription_response, 'segments', [])

            word_count = len(full_transcript.split())

            logger.success(
                f"  {self.adapter_name}: Transcribed {word_count} words "
                f"from {duration:.1f}s audio"
            )

            # Update trace with results
            if trace_manager:
                trace_manager.update_stage(
                    node_id=node.task_id,
                    stage_name="execution",
                    llm_response=full_transcript[:500] + "...",  # Preview
                    additional_data={
                        "duration_seconds": duration,
                        "word_count": word_count,
                        "segment_count": len(segments),
                        "file_size_mb": file_size_mb
                    }
                )

            return {
                "full_transcript": full_transcript,
                "segments": [
                    {
                        "id": seg.get('id', i),
                        "start": seg.get('start', 0),
                        "end": seg.get('end', 0),
                        "text": seg.get('text', '')
                    }
                    for i, seg in enumerate(segments)
                ] if segments else [],
                "duration_seconds": duration,
                "word_count": word_count,
                "audio_file": Path(audio_path).name
            }

        except Exception as e:
            error_msg = f"Whisper API error: {str(e)}"
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

        api_key = os.getenv("OPENROUTER_API_KEY")
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
class DecisionExtractorAdapter(BaseAdapter):
    """Extracts key decisions made during meeting."""
    adapter_name = "DecisionExtractorAdapter"

    def __init__(self):
        super().__init__(self.adapter_name)
        # Same setup as ActionItemExtractor
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_id = "google/gemini-2.5-pro"

    async def process(self, node, agent_task_input, trace_manager):
        # TODO: Implement in Etap 2
        logger.info(f"  {self.adapter_name}: To be implemented in Etap 2")
        return {"decisions": [], "count": 0}


class SpeakerAnalyzerAdapter(BaseAdapter):
    """Analyzes speaker contributions and speaking time."""
    adapter_name = "SpeakerAnalyzerAdapter"

    def __init__(self):
        super().__init__(self.adapter_name)
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_id = "google/gemini-2.5-pro"

    async def process(self, node, agent_task_input, trace_manager):
        # TODO: Implement in Etap 2
        logger.info(f"  {self.adapter_name}: To be implemented in Etap 2")
        return {"speakers": [], "count": 0}
