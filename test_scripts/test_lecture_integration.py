#!/usr/bin/env python3
"""
ETAP 4 Test: Educational Lecture Processing

Tests lecture-specific processing with:
- ProfiledSentientAgent + educational_lecture_agent profile
- LecturePlanner (autonomous extraction)
- Parallel chunking for large lecture files (644MB)
- Educational content extraction (concepts, formulas, code, exam hints)
- Study guide generation

Usage:
    python3 test_scripts/test_lecture_integration.py

Environment required:
    - OPENAI_API_KEY (for Whisper)
    - OPENROUTER_API_KEY (for extractors)
    - .venv activated

Output:
    test_output/etap4_lecture_execution.json
    Console logs showing lecture processing
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sentientresearchagent.framework_entry import ProfiledSentientAgent


def test_lecture_processing():
    """Test full ROMA execution with lecture profile."""

    print("\n" + "=" * 70)
    print("🚀 ETAP 4: Educational Lecture Processing Test")
    print("=" * 70)

    # Find lecture file
    audio_dir = Path(__file__).parent.parent / "test_transcript"

    # Look for lecture file (second .mp4 file - IZBDwz_AE9I)
    audio_files = sorted(list(audio_dir.glob("*.mp4")))

    if len(audio_files) < 2:
        print("\n❌ Lecture file not found in test_transcript/")
        print(f"   Found only {len(audio_files)} file(s)")
        sys.exit(1)

    # Use second file (lecture)
    audio_path = str(audio_files[1])

    print(f"\n📁 Test Lecture:")
    print(f"   Path: {audio_path}")
    print(f"   Name: {Path(audio_path).name}")

    file_size_mb = os.path.getsize(audio_path) / (1024 ** 2)
    print(f"   Size: {file_size_mb:.1f} MB")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("\n❌ OPENAI_API_KEY not set!")
        sys.exit(1)

    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        print("\n❌ OPENROUTER_API_KEY not set!")
        sys.exit(1)

    print(f"\n✅ API keys configured")

    # Initialize ROMA with lecture profile
    print(f"\n🔧 Initializing ROMA Framework...")
    print(f"   Profile: educational_lecture_agent")

    try:
        agent = ProfiledSentientAgent.create_with_profile(
            profile_name="educational_lecture_agent",
            enable_hitl_override=False,
            max_planning_depth=3
        )

        print(f"   ✅ Agent initialized")
        print(f"   ✅ TaskGraph ready")
        print(f"   ✅ KnowledgeStore ready")

    except Exception as e:
        print(f"\n❌ Failed to initialize ROMA agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Prepare execution goal
    goal = f"""Process this educational lecture recording and generate comprehensive study materials.

Lecture file: {Path(audio_path).name}
Type: Educational lecture
Duration: ~65 minutes

Extract:
- Key concepts and definitions
- Mathematical formulas and equations
- Code examples (if any)
- Exam preparation hints
- Important topics flagged by professor

Generate comprehensive study guide for exam preparation."""

    # Pass audio path via environment (same as standup test)
    os.environ['TEMP_AUDIO_PATH'] = audio_path

    print(f"\n🚀 Starting ROMA Execution...")
    print(f"   Goal: {goal[:80]}...")
    print(f"   Max steps: 200 (lecture is longer)")
    print(f"\n⏳ Expected processing time: 3-5 minutes")
    print(f"   - Transcription: ~2-3 min ({file_size_mb:.0f}MB, ~7 chunks)")
    print(f"   - Extraction: ~1-2 min (parallel)")
    print(f"   - Study guide: ~30-60s")
    print()

    start_time = datetime.now()

    try:
        # Execute ROMA
        result = agent.execute(
            goal=goal,
            max_steps=200
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Display results
        print("\n" + "=" * 70)
        print("✅ ROMA EXECUTION COMPLETE")
        print("=" * 70)

        print(f"\n📊 Execution Statistics:")
        print(f"   Status: {result['status']}")
        print(f"   Total time: {result['execution_time']:.1f}s ({result['execution_time']/60:.1f} min)")
        print(f"   Nodes created: {result['node_count']}")

        # Analyze structure
        task_graph = agent.task_graph

        print(f"\n🌳 Task Graph Structure:")
        print(f"   Total graphs: {len(task_graph.graphs)}")
        print(f"   Total nodes: {len(task_graph.nodes)}")

        # Show hierarchy
        print(f"\n📂 Node Hierarchy:")
        nodes_by_layer = {}
        for node_id, node in task_graph.nodes.items():
            layer = node.layer
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(node)

        for layer in sorted(nodes_by_layer.keys()):
            nodes = nodes_by_layer[layer]
            print(f"\n   Layer {layer}: ({len(nodes)} nodes)")
            for node in nodes[:10]:  # First 10
                indent = "    " * (layer + 1)
                status_icon = "✅" if node.status.name == "DONE" else "⏳"
                print(f"{indent}{status_icon} [{node.task_id}] {node.node_type.name}")
                print(f"{indent}   Goal: {node.goal[:60]}...")

        # Check final output
        print(f"\n📤 Final Output (Study Guide Preview):")

        final_result = result.get('final_output') or result.get('final_result')

        if final_result:
            output_preview = str(final_result)[:800]
            print(f"{output_preview}...")
        else:
            print("   (No final output in result)")

        # Save results
        output_dir = Path(__file__).parent.parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        result_file = output_dir / "etap4_lecture_execution.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n💾 Execution result saved to: {result_file}")

        # Validation checks
        print(f"\n🧪 Validation Checks:")
        checks_passed = 0
        checks_total = 6

        if result['status'] == 'completed':
            print("   ✅ Execution completed successfully")
            checks_passed += 1
        else:
            print(f"   ❌ Execution status: {result['status']}")

        if result['node_count'] >= 6:
            print(f"   ✅ Expected node structure (>= 6 nodes)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Only {result['node_count']} nodes")

        if len(nodes_by_layer) >= 2:
            print(f"   ✅ Multi-layer execution ({len(nodes_by_layer)} layers)")
            checks_passed += 1
        else:
            print(f"   ❌ Only {len(nodes_by_layer)} layer(s)")

        # Check for parallel extraction (Layer 1 should have 4+ extractors)
        layer_1_nodes = nodes_by_layer.get(1, [])
        if len(layer_1_nodes) >= 5:
            print(f"   ✅ Parallel extraction ({len(layer_1_nodes)} Layer-1 nodes)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Limited parallelization ({len(layer_1_nodes)} nodes)")

        if result['execution_time'] < 600:  # 10 minutes
            print(f"   ✅ Performance acceptable (<10 min)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Slow execution ({result['execution_time']:.1f}s)")

        # Check if study guide contains educational content
        output_str = str(final_result).lower()
        has_educational_content = any(
            keyword in output_str
            for keyword in ['concept', 'formula', 'study', 'exam', 'lecture']
        )
        if has_educational_content:
            print(f"   ✅ Study guide contains educational content")
            checks_passed += 1
        else:
            print(f"   ⚠️  Study guide missing educational keywords")

        # Final verdict
        print("\n" + "=" * 70)
        if checks_passed >= 5:
            print("🎉 ETAP 4 TEST: PASSED")
            print(f"   ✅ {checks_passed}/{checks_total} checks passed")
            print("\n➡️  Lecture Processing Successful!")
            print("➡️  Ready for production deployment")
            return_code = 0
        else:
            print(f"⚠️  ETAP 4 TEST: PARTIAL ({checks_passed}/{checks_total} checks)")
            print("   Review execution logs above")
            return_code = 1

        print("=" * 70)

        sys.exit(return_code)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during ROMA execution:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n🚀 Starting ETAP 4: Educational Lecture Processing Test\n")
    test_lecture_processing()
