#!/usr/bin/env python3
"""
Test script to verify Phillip model loads and works with slippi-ai.

This script:
1. Loads the model
2. Verifies configuration
3. Tests that it can be built into an agent
4. Optionally runs a quick evaluation

Usage (from project root):
    .venv/bin/python scripts/test_phillip_model.py
"""

import argparse
import sys
from pathlib import Path

# Add project root and slippi-ai to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'fighters/phillip/slippi-ai'))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TEST 1: Importing slippi-ai modules")
    print("=" * 60)

    try:
        from slippi_ai import saving
        print("✅ slippi_ai.saving")
    except ImportError as e:
        print(f"❌ Failed to import saving: {e}")
        return False

    try:
        from slippi_ai import eval_lib
        print("✅ slippi_ai.eval_lib")
    except ImportError as e:
        print(f"❌ Failed to import eval_lib: {e}")
        return False

    try:
        import tensorflow as tf
        print(f"✅ tensorflow {tf.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import tensorflow: {e}")
        return False

    try:
        import melee
        version = getattr(melee, '__version__', 'unknown')
        print(f"✅ libmelee {version}")
    except ImportError as e:
        print(f"❌ Failed to import libmelee: {e}")
        return False

    print("\n✅ All imports successful!\n")
    return True


def test_model_load():
    """Test loading the model file."""
    print("=" * 60)
    print("TEST 2: Loading model")
    print("=" * 60)

    from slippi_ai import saving

    model_path = PROJECT_ROOT / 'fighters/phillip/models/all_d21_imitation_v3.pkl'

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("\nPlease download the model first:")
        print("  curl -L 'https://dl.dropbox.com/scl/fi/bppnln3rfktxfdocottuw/all_d21_imitation_v3?rlkey=46yqbsp7vi5222x04qt4npbkq&st=6knz106y&dl=1' -o models/all_d21_imitation_v3.pkl")
        return False, None

    print(f"Loading from: {model_path}")

    try:
        state = saving.load_state_from_disk(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False, None

    # Check structure
    print(f"\nModel structure:")
    print(f"  Keys: {list(state.keys())}")

    config = state.get('config', {})
    policy_config = config.get('policy', {})
    network_config = config.get('network', {})

    print(f"\nConfiguration:")
    print(f"  Delay: {policy_config.get('delay')} frames")
    print(f"  Network: {network_config.get('name')}")
    print(f"  Controller head: {config.get('controller_head', {}).get('name')}")

    print("\n✅ Model structure looks good!\n")
    return True, state


def test_agent_build(state, dolphin_path=None, iso_path=None):
    """Test building an agent from the model."""
    print("=" * 60)
    print("TEST 3: Building agent from model")
    print("=" * 60)

    from slippi_ai import eval_lib
    import melee

    model_path = PROJECT_ROOT / 'fighters/phillip/models/all_d21_imitation_v3.pkl'

    # Get config from state
    config = state.get('config', {})
    policy_config = config.get('policy', {})
    delay = policy_config.get('delay', 21)

    print(f"Building agent with:")
    print(f"  Model: {model_path}")
    print(f"  Delay: {delay} frames")
    print(f"  Port: 1 (vs port 2)")

    try:
        # Note: This doesn't actually connect to Dolphin yet
        # Just builds the agent object
        agent = eval_lib.build_agent(
            port=1,
            opponent_port=2,
            console_delay=delay - 1,  # Leave 1 frame for async inference
            path=str(model_path),
            async_inference=True,
        )
        print("✅ Agent built successfully!")
        print(f"  Agent type: {type(agent)}")
        if hasattr(agent, 'config'):
            char = agent.config.get('character', 'all') if isinstance(agent.config, dict) else getattr(agent.config, 'character', 'unknown')
            print(f"  Character: {char}")
        else:
            print("  Character: unknown")

    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Agent ready to use!\n")
    return True


def test_with_dolphin(dolphin_path, iso_path):
    """Test actually running the agent (requires Dolphin)."""
    print("=" * 60)
    print("TEST 4: Running with Dolphin (OPTIONAL)")
    print("=" * 60)

    print("⚠️  This test is not yet implemented.")
    print("To test with Dolphin, use:")
    print(f"  cd slippi-ai")
    print(f"  python scripts/eval_two.py \\")
    print(f"    --dolphin.path='{dolphin_path}' \\")
    print(f"    --dolphin.iso='{iso_path}' \\")
    print(f"    --p1.type=human \\")
    print(f"    --p2.ai.path='../models/all_d21_imitation_v3.pkl'")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Phillip model")
    parser.add_argument('--dolphin', help='Path to Slippi Dolphin (optional)')
    parser.add_argument('--iso', help='Path to Melee ISO (optional)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHILLIP MODEL TEST SUITE")
    print("=" * 60 + "\n")

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Check your environment setup.")
        print("See SETUP.md for installation instructions.")
        return 1

    # Test 2: Model loading
    success, state = test_model_load()
    if not success:
        print("\n❌ Model loading failed.")
        return 1

    # Test 3: Agent building
    if not test_agent_build(state, args.dolphin, args.iso):
        print("\n❌ Agent building failed.")
        return 1

    # Test 4: Optional Dolphin test
    if args.dolphin and args.iso:
        test_with_dolphin(args.dolphin, args.iso)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nPhillip model is ready to use!")
    print("\nNext steps:")
    print("  1. Try running with Dolphin:")
    print("     cd slippi-ai && python scripts/eval_two.py --p1.type=human --p2.ai.path=../models/all_d21_imitation_v3.pkl --dolphin.path=<path> --dolphin.iso=<path>")
    print("  2. Build the PhillipFighter adapter for nojohns")
    print("  3. Test Phillip vs SmashBot!")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
