#!/usr/bin/env python3
"""Verify llmbeans package imports and CLI entry points."""

import sys


def test_imports():
    print("Testing imports...")

    try:
        from llmbeans.cli import main, generate_summary, write_scripts
        from llmbeans.models.scanner import ModelInfo, scan
        from llmbeans.hardware.profiles import HardwareProfileEntry, load_profiles
        from llmbeans.hardware.detector import detect_hardware, HardwareProfile
        from llmbeans.recommenders.engine import recommend, Recommendation, get_available_tools
        from llmbeans.output.script_gen import generate_shell_script, generate_batch_script
        print("✓ llmbeans modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import llmbeans modules: {e}")
        return False

    assert callable(main)
    assert callable(generate_summary)
    assert callable(write_scripts)
    assert load_profiles()
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("llmbeans Import Test")
    print("=" * 50)
    ok = test_imports()
    print("=" * 50)
    if ok:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    print("✗ SOME TESTS FAILED")
    sys.exit(1)
