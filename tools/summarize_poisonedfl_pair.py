"""Summarize paired PoisonedFL JSONL files without project dependencies."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--base", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.logs.glob(f"*_poisonedfl_pair_*.metrics.jsonl")):
        try:
            run_id = int(path.name.split("_", 1)[0])
        except ValueError:
            continue
        if not args.base < run_id <= args.base + 8:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["source"] = path.name
                rows.append(row)
    if not rows:
        raise SystemExit(f"No metrics found for base {args.base} in {args.logs}")
    conditions = {}
    for row in rows:
        conditions.setdefault(row["condition"], []).append(row)
    fields = ["condition", "strength", "spc_enabled", "rounds", "last_round",
              "final_server_acc", "final_ensemble_acc", "mean_last10_server_acc",
              "mean_last10_aggregate_shift_rms", "mean_last10_raw_pre_filter_aggregate_shift_rms",
              "spc_tp", "spc_fp", "spc_fn", "spc_tn", "complete"]
    output = []
    for condition, data in sorted(conditions.items()):
        tail = data[-10:]
        final = data[-1]
        expected_last = max(int(r["round"]) for r in rows)
        output.append({
            "condition": condition, "strength": final.get("strength"),
            "spc_enabled": final.get("spc_enabled", False), "rounds": len(data),
            "last_round": final["round"], "final_server_acc": final["server_test_acc"],
            "final_ensemble_acc": final["ensemble_test_acc"],
            "mean_last10_server_acc": sum(r["server_test_acc"] for r in tail) / len(tail),
            "mean_last10_aggregate_shift_rms": sum(r.get("aggregate_shift_rms", 0) for r in tail) / len(tail),
            "mean_last10_raw_pre_filter_aggregate_shift_rms": sum(
                r.get("raw_pre_filter_aggregate_shift_rms", 0) for r in tail) / len(tail),
            "spc_tp": sum(r.get("spc_tp", 0) for r in tail),
            "spc_fp": sum(r.get("spc_fp", 0) for r in tail),
            "spc_fn": sum(r.get("spc_fn", 0) for r in tail),
            "spc_tn": sum(r.get("spc_tn", 0) for r in tail),
            "complete": int(final["round"]) == expected_last,
        })
    destination = args.output or args.logs / f"{args.base}_poisonedfl_paired70_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {destination}")
    for row in output:
        print("{condition}: round={last_round}, final={final_server_acc:.6f}, SPC={spc_enabled}, complete={complete}".format(**row))


if __name__ == "__main__":
    main()
