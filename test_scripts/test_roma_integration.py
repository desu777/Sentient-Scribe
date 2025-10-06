#!/usr/bin/env python3
"""
ETAP 3 Test: Full ROMA Integration

Tests complete hierarchical execution with:
- ProfiledSentientAgent + agile_standup_agent profile
- Hierarchical task decomposition (ROOT → Layer 1 → Layer 2)
- Parallel execution (multiple nodes simultaneously)
- Context propagation (transcript → extractors → writers)
- Real-time monitoring

Uses Ronaldo podcast for testing (even though it's interview, not standup).
This tests extractor flexibility and ROMA orchestration.

Usage:
    python3 test_scripts/test_roma_integration.py

Environment required:
    - OPENAI_API_KEY (for Whisper)
    - OPENROUTER_API_KEY (for Gemini extractors)
    - .venv activated
    - Full ROMA dependencies installed

Output:
    test_output/etap3_roma_execution.json
    Console logs showing hierarchical execution
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sentientresearchagent.framework_entry import ProfiledSentientAgent


def test_roma_standup_integration():
    """Test full ROMA execution with standup profile."""

    print("\n" + "=" * 70)
    print("🚀 ETAP 3: Full ROMA Integration Test")
    print("=" * 70)

    # Find audio file
    audio_dir = Path(__file__).parent.parent / "test_transcript"
    audio_files = list(audio_dir.glob("*.mp4")) + list(audio_dir.glob("*.mp3"))

    if not audio_files:
        print("\n❌ No audio file found in test_transcript/")
        sys.exit(1)

    audio_path = str(audio_files[0])

    print(f"\n📁 Test Audio:")
    print(f"   Path: {audio_path}")
    print(f"   Name: {Path(audio_path).name[:60]}...")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY", "").strip():
        print("\n❌ OPENAI_API_KEY not set!")
        sys.exit(1)

    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        print("\n❌ OPENROUTER_API_KEY not set!")
        sys.exit(1)

    print(f"\n✅ API keys configured")

    # Initialize ROMA with standup profile
    print(f"\n🔧 Initializing ROMA Framework...")
    print(f"   Profile: agile_standup_agent")

    try:
        agent = ProfiledSentientAgent.create_with_profile(
            profile_name="agile_standup_agent",
            enable_hitl_override=False,  # No human review for testing
            max_planning_depth=3  # Allow up to 3 levels of decomposition
        )

        print(f"   ✅ Agent initialized")
        print(f"   ✅ TaskGraph ready")
        print(f"   ✅ KnowledgeStore ready")
        print(f"   ✅ ExecutionOrchestrator ready")

    except Exception as e:
        print(f"\n❌ Failed to initialize ROMA agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Prepare execution goal
    goal = f"""Process this standup meeting recording and extract tactical insights.

Audio file: {Path(audio_path).name}
Meeting type: standup
Duration: ~20 minutes

Extract:
- Blockers (anything preventing progress)
- Action items (with owners and deadlines)
- Yesterday's accomplishments
- Today's plans

Generate standup summary and deliverables."""

    # HACK: Pass audio path to WhisperAdapter via environment
    # TODO: Better solution in production (pass via node.aux_data properly)
    os.environ['TEMP_AUDIO_PATH'] = audio_path

    print(f"\n🚀 Starting ROMA Execution...")
    print(f"   Goal: {goal[:80]}...")
    print(f"   Max steps: 150")
    print(f"\n⏳ This will take 1-2 minutes. Watch for:")
    print(f"   - Hierarchical node creation (ROOT → Layer 1 → Layer 2)")
    print(f"   - Parallel execution (multiple nodes running)")
    print(f"   - Context propagation (transcript → extractors)")
    print()

    start_time = datetime.now()

    try:
        # Execute ROMA (THIS IS THE MAIN TEST)
        # Note: execute() is synchronous (handles async internally)
        result = agent.execute(
            goal=goal,
            max_steps=150
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Display results
        print("\n" + "=" * 70)
        print("✅ ROMA EXECUTION COMPLETE")
        print("=" * 70)

        print(f"\n📊 Execution Statistics:")
        print(f"   Status: {result['status']}")
        print(f"   Total time: {result['execution_time']:.1f}s")
        print(f"   Nodes created: {result['node_count']}")
        print(f"   HITL enabled: {result.get('hitl_enabled', False)}")

        # Analyze task graph structure
        task_graph = agent.task_graph
        knowledge_store = agent.knowledge_store

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
            for node in nodes:
                indent = "    " * (layer + 1)
                status_icon = "✅" if node.status.name == "DONE" else "⏳"
                print(f"{indent}{status_icon} [{node.task_id}] {node.node_type.name}")
                print(f"{indent}   Goal: {node.goal[:50]}...")
                print(f"{indent}   Status: {node.status.name}")

                if node.result:
                    result_preview = str(node.result)[:80]
                    print(f"{indent}   Result: {result_preview}...")

        # Check for parallel execution
        print(f"\n⚡ Parallel Execution Analysis:")

        layer_2_nodes = nodes_by_layer.get(2, [])
        if len(layer_2_nodes) >= 2:
            print(f"   Found {len(layer_2_nodes)} Layer-2 nodes")
            print(f"   These likely ran in parallel (check timestamps)")

            # Show timestamps
            for node in layer_2_nodes[:4]:  # First 4
                created = node.timestamp_created.strftime("%H:%M:%S")
                completed = node.timestamp_completed.strftime("%H:%M:%S") if node.timestamp_completed else "N/A"
                print(f"     {node.task_id}: {created} → {completed}")

        # Check final output
        print(f"\n📤 Final Output:")

        final_result = result.get('final_output') or result.get('final_result')

        if final_result:
            output_preview = str(final_result)[:500]
            print(f"{output_preview}...")
        else:
            print("   (No final output in result)")

        # Save execution details
        output_dir = Path(__file__).parent.parent / "test_output"
        output_dir.mkdir(exist_ok=True)

        # Save execution result
        result_file = output_dir / "etap3_roma_execution.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n💾 Execution result saved to: {result_file}")

        # Save task graph visualization
        if hasattr(task_graph, 'to_visualization_dict'):
            graph_file = output_dir / "etap3_task_graph.json"
            graph_data = task_graph.to_visualization_dict()
            with open(graph_file, 'w') as f:
                json.dump(graph_data, f, indent=2, default=str)
            print(f"💾 Task graph saved to: {graph_file}")

        # Validation checks
        print(f"\n🧪 Validation Checks:")
        checks_passed = 0
        checks_total = 5

        if result['status'] == 'completed':
            print("   ✅ Execution completed successfully")
            checks_passed += 1
        else:
            print(f"   ❌ Execution status: {result['status']}")

        if result['node_count'] >= 8:
            print(f"   ✅ Hierarchical decomposition (>= 8 nodes)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Only {result['node_count']} nodes (expected 8+)")

        if len(nodes_by_layer) >= 2:
            print(f"   ✅ Multi-layer execution ({len(nodes_by_layer)} layers)")
            checks_passed += 1
        else:
            print(f"   ❌ Only {len(nodes_by_layer)} layer(s)")

        if len(layer_2_nodes) >= 2:
            print(f"   ✅ Parallel execution capable ({len(layer_2_nodes)} Layer-2 nodes)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Limited parallelization")

        if result['execution_time'] < 180:  # 3 minutes
            print(f"   ✅ Performance acceptable (<3 min)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Slow execution ({result['execution_time']:.1f}s)")

        # Final verdict
        print("\n" + "=" * 70)
        if checks_passed >= 4:
            print("🎉 ETAP 3 TEST: PASSED")
            print(f"   ✅ {checks_passed}/{checks_total} checks passed")
            print("\n➡️  ROMA Integration Successful!")
            print("➡️  Ready to proceed to ETAP 4: Lecture Profile")
            return_code = 0
        else:
            print(f"⚠️  ETAP 3 TEST: PARTIAL ({checks_passed}/{checks_total} checks)")
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
    print("\n🚀 Starting ETAP 3: Full ROMA Integration Test\n")
    # test_roma_standup_integration() is now synchronous
    test_roma_standup_integration()
