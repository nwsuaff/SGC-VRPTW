"""Generate shell commands to run each instance as a separate process.

Usage:
    python gen_batch.py                    # list all commands
    python gen_batch.py --instances 5      # list commands for first 5 instances
    python gen_batch.py --dry-run         # show commands without saving
    python gen_batch.py --save            # save commands to a .txt file
"""
import argparse
from pathlib import Path


def gen_commands(
    data_dir: str = "data/VRPTW/GH800",
    instances_filter: str = "gh800",
    llm: str = "openai",
    solver: str = "ortools",
    max_iterations: int = 2,
    per_instance_timeout: int = 900,
    output_dir: str = "results/gh800",
    max_instances: int = None,
):
    data_path = Path(data_dir)
    all_vrp_files = sorted(data_path.glob("*.vrp"))

    def get_size(filename: str) -> int:
        import re
        match = re.search(r"_(\d+)_", filename)
        return int(match.group(1)) * 100 if match else 0

    if instances_filter == "all":
        filtered = sorted(all_vrp_files)
    else:
        target = int(instances_filter.replace("gh", ""))
        filtered = sorted(f for f in all_vrp_files if get_size(f.name) == target)

    if max_instances:
        filtered = filtered[:max_instances]

    lines = []
    for f in filtered:
        cmd = (
            f'python -m src.cli droc "{f}" --llm {llm} '
            f'--max-iterations {max_iterations} --time-limit {per_instance_timeout} '
            f'--output-dir "{output_dir}/{f.stem}"'
        )
        lines.append(cmd)

    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-instance droc commands")
    parser.add_argument("--data-dir", default="data/VRPTW/GH800")
    parser.add_argument("--instances", default="gh800",
                        help="gh200, gh400, gh800, gh1000, all")
    parser.add_argument("--llm", default="openai")
    parser.add_argument("--solver", default="ortools")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--per-instance-timeout", type=int, default=900)
    parser.add_argument("--output-dir", default="results/gh800")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="show commands without saving")
    parser.add_argument("--save", action="store_true",
                        help="save commands to gen_batch_commands.txt")
    parser.add_argument("--count", action="store_true",
                        help="only print count and exit")
    args = parser.parse_args()

    cmds = gen_commands(
        data_dir=args.data_dir,
        instances_filter=args.instances,
        llm=args.llm,
        solver=args.solver,
        max_iterations=args.max_iterations,
        per_instance_timeout=args.per_instance_timeout,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
    )

    if args.count:
        print(f"{len(cmds)} commands")
        print(f"Estimated time: ~{len(cmds) * 4 / 60:.0f} min (at 4min/instance)")
        print(f"Recommended parallel runs: 2-4")
        exit(0)

    print(f"Generated {len(cmds)} commands:\n")
    for i, cmd in enumerate(cmds, 1):
        print(f"[{i}/{len(cmds)}] {cmd}")

    if args.save:
        path = Path("gen_batch_commands.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(cmds) + "\n")
        print(f"\nSaved to {path}")

    print(f"\nRun commands in parallel terminals, e.g.:")
    print(f"  Split your terminal into 4 panes, paste commands 1-4 in pane 1,")
    print(f"  5-8 in pane 2, etc.  Or use a job runner.")
