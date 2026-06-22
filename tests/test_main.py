"""Tests for llmbeans.main entry point."""

import importlib
import sys
from unittest.mock import patch

import pytest


def test_main_delegates_to_cli():
    with patch("llmbeans.cli.main") as mock_cli:
        from llmbeans.main import main
        main()
        mock_cli.assert_called_once()


def test_main_import_error(capsys):
    import llmbeans.main as main_module

    real_import = builtins.__import__ if False else __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llmbeans.cli":
            raise ImportError("missing rich")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=guarded_import):
        with pytest.raises(SystemExit) as exc:
            main_module.main()
    assert exc.value.code == 1


def test_main_keyboard_interrupt(capsys):
    import llmbeans.main as main_module

    with patch("llmbeans.cli.main", side_effect=KeyboardInterrupt()):
        with pytest.raises(SystemExit) as exc:
            main_module.main()
    assert exc.value.code == 0


def test_main_module_name_guard():
    import llmbeans.main as main_module
    with patch.object(main_module, "main") as mock_main:
        exec(compile("if __name__ == '__main__': main()", "llmbeans/main.py", "exec"), {"__name__": "__main__", "main": mock_main})
        mock_main.assert_called_once()
