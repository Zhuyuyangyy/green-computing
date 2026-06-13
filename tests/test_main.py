"""Tests for main.py CLI entry point."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMainCLI:
    def test_import_main(self):
        import main
        assert hasattr(main, "main")
        assert hasattr(main, "demo_mode")
        assert hasattr(main, "train_mode")
        assert hasattr(main, "evaluate_mode")

    def test_print_banner(self, capsys):
        from main import print_banner
        print_banner()
        captured = capsys.readouterr()
        assert "Green Computing" in captured.out

    def test_demo_mode(self, capsys):
        from main import demo_mode
        demo_mode()
        captured = capsys.readouterr()
        assert "SAC" in captured.out
        assert "TD3" in captured.out
        assert "PUE" in captured.out
        assert "1.18" in captured.out

    def test_evaluate_mode(self, capsys):
        from main import evaluate_mode
        evaluate_mode()
        captured = capsys.readouterr()
        assert "PUE" in captured.out

    def test_main_no_args(self, capsys):
        from main import main
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["main.py"]
            main()
            captured = capsys.readouterr()
            assert "Green Computing" in captured.out
        finally:
            sys.argv = sys_argv_backup

    def test_main_demo(self, capsys):
        from main import main
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["main.py", "demo"]
            main()
            captured = capsys.readouterr()
            assert "演示模式" in captured.out
        finally:
            sys.argv = sys_argv_backup

    def test_main_evaluate(self, capsys):
        from main import main
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["main.py", "evaluate"]
            main()
            captured = capsys.readouterr()
            assert "评估模式" in captured.out
        finally:
            sys.argv = sys_argv_backup

    def test_train_mode_missing_deps(self, capsys, monkeypatch):
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from main import train_mode
        train_mode(1)
        captured = capsys.readouterr()
        assert "错误" in captured.out or "依赖" in captured.out
