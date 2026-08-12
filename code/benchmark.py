from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMMANDS = {
    "validate": "validate_all.py",
    "generate": "generate_all.py",
    "evaluate": "evaluate_all.py",
    "report": "report_all.py",
}


def print_usage() -> None:
    print("Visual Reasoning Benchmark Suite")
    print("\nCommands:")
    print("  python benchmark.py validate [args]")
    print("  python benchmark.py generate [args]")
    print("  python benchmark.py evaluate [args]")
    print("  python benchmark.py report   [args]")
    print("  python benchmark.py build-nonogram [args]")
    print("  python benchmark.py build-tangram  [args]")
    print("\nUse 'python benchmark.py <command> --help' for details.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print_usage()
        return
    command = sys.argv[1]
    if command in {"build-nonogram", "build-tangram"}:
        tool_name = "build_nonogram_benchmark.py" if command == "build-nonogram" else "build_tangram_benchmark.py"
        dataset_name = "nonogram" if command == "build-nonogram" else "tangram"
        script = Path(__file__).resolve().parent / "tools" / tool_name
        extra = list(sys.argv[2:])
        if "--output" not in extra:
            suite_root = Path(__file__).resolve().parents[1]
            extra = ["--output", str(suite_root / "datasets" / dataset_name), *extra]
        raise SystemExit(subprocess.call([sys.executable, str(script), *extra]))
    if command == "list":
        from task_registry import TASKS
        total = 0
        for name, spec in TASKS.items():
            print(f"{name:28} {spec.expected_count:4}  {spec.display_name}")
            total += spec.expected_count
        print(f"{'TOTAL':28} {total:4}")
        return
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n")
        print_usage()
        raise SystemExit(2)
    script = Path(__file__).resolve().parent / COMMANDS[command]
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[2:]]))


if __name__ == "__main__":
    main()
