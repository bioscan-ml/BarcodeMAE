"""
Audit tool for its_family_ablation_knn.sh's SLURM array.

The current version of that script runs one (arch, aux, representation)
combination per array task -- 18 tasks total (2 archs x 3 aux objectives x 3
representations). An older version of the script looped over all 3
representations sequentially inside one task (6 array tasks); this tool
still reads logs from that older layout too, since both write results to
the same shared file results_final/KNN_ITS_family_k1.txt with the same key
format, and old logs may still be lying around.

Two distinct reasons a (arch, aux, repr) result can be missing:

  1. It genuinely never finished: the task's log has no matching entry at
     all, or ends abruptly mid-run (most likely a SLURM walltime kill --
     each representation requires a full pass over the 5.23M-sequence
     gallery to embed it, which is why the old sequential-loop version of
     this script routinely got killed partway through its 2nd or 3rd
     representation before the time limit).
  2. It finished and printed a result, but the result-file line was lost:
     knn_its_clean.py writes each (task,k) result to the shared results
     file in the same loop iteration it's printed in, but many array tasks
     append to that ONE file concurrently -- on a parallel filesystem
     without locking, two simultaneous appends can silently drop a line
     even though the value was correctly computed.

This script tells the two cases apart: for every (arch, aux, repr) in the
full 18-entry grid it reports whether a log exists, whether that log's run
finished cleanly, and cross-checks every accuracy value actually printed to
stdout against the results file, flagging any that are printed but absent
from the file (case 2 -- recoverable without rerunning anything).

Run on the cluster (standard library only):

    python barcodebert/check_its_family_ablation_logs.py \\
        --results-file results_final/KNN_ITS_family_k1.txt

--logs-dir defaults to final_logs/ and is searched recursively, so it does
not need to be pointed at one specific array job's folder -- logs from
other, unrelated jobs mixed into that tree are silently skipped.
"""
import argparse
import glob
import os
import re

ARCHES = ["maelm", "transformer"]
AUX_TASKS = ["binary", "triplet", "ce"]
REPRS = ["tokens", "cls", "tokens_with_cls"]
TEST_SET_TAGS = {
    "Test1 (Yeast)": "test1",
    "Test2 (Filamentous)": "test2",
    "Test3 (MycoAI)": "test3",
}

# New (current) script version: one repr per task, printed directly in the header.
HEADER_RE_NEW = re.compile(
    r"^Arch: (?P<arch>\S+) \| Aux: (?P<aux>\S+) \| Repr: (?P<repr>\S+) \| Ckpt: (?P<ckpt>\S+)"
)
# Old script version: 3 reprs looped per task, repr not in the header --
# recovered instead from each sequential run's "Namespace(...)" printout.
HEADER_RE_OLD = re.compile(r"^Arch: (?P<arch>\S+) \| Aux: (?P<aux>\S+) \| Ckpt: (?P<ckpt>\S+)")

CKPT_MISSING_RE = re.compile(r"^ERROR: checkpoint not found at (\S+)")
REPR_FAILED_RE = re.compile(r"^ERROR: knn_its_clean\.py failed for (\S+)")
ALL_DONE_RE = re.compile(r"^All done at: .* \| exit: (\d+)")
RUN_NAME_IN_NAMESPACE_RE = re.compile(r"run_name='([^']*)'")
TESTSET_START_RE = re.compile(r"^(Test\d \([^)]+\)): (\d+) query specimens")
TASK_START_RE = re.compile(r"^\s*--- (\S+) ---")
ACCURACY_RE = re.compile(r"^\s*\[(\S+)\] k=(\d+): accuracy=([\d.]+)%")


def parse_log(path):
    """Returns a dict with: path, arch, aux, ckpt, ckpt_missing, exit_code,
    repr_failures (set of reprs explicitly reported as failed), and runs
    (list of {"repr": str|None, "accuracies": [(test_tag, task, k, acc), ...]})."""
    info = {
        "path": path,
        "arch": None,
        "aux": None,
        "ckpt": None,
        "ckpt_missing": False,
        "exit_code": None,
        "repr_failures": set(),
        "runs": [],
    }
    current_run = None
    current_test_tag = None
    current_task = None
    header_format = None  # "new" | "old"

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if info["arch"] is None:
                m = HEADER_RE_NEW.match(line)
                if m:
                    info["arch"], info["aux"], info["ckpt"] = m.group("arch"), m.group("aux"), m.group("ckpt")
                    header_format = "new"
                    current_run = {"repr": m.group("repr"), "accuracies": []}
                    info["runs"].append(current_run)
                    continue
                m = HEADER_RE_OLD.match(line)
                if m:
                    info["arch"], info["aux"], info["ckpt"] = m.group("arch"), m.group("aux"), m.group("ckpt")
                    header_format = "old"
                    continue

            if CKPT_MISSING_RE.match(line):
                info["ckpt_missing"] = True
                continue

            m = REPR_FAILED_RE.match(line)
            if m:
                info["repr_failures"].add(m.group(1))
                continue

            m = ALL_DONE_RE.match(line)
            if m:
                info["exit_code"] = int(m.group(1))
                continue

            if header_format == "old":
                # A new "Configuration:\n\nNamespace(...)" block marks the start
                # of one of the (up to 3) sequential knn_its_clean.py invocations.
                m = RUN_NAME_IN_NAMESPACE_RE.search(line)
                if m and line.strip().startswith("Namespace("):
                    run_name = m.group(1)
                    repr_ = None
                    for r in REPRS:
                        if run_name.endswith(f"_{r}"):
                            repr_ = r
                            break
                    current_run = {"repr": repr_, "accuracies": []}
                    info["runs"].append(current_run)
                    current_test_tag = None
                    current_task = None
                    continue

            m = TESTSET_START_RE.match(line)
            if m:
                current_test_tag = TEST_SET_TAGS.get(m.group(1))
                current_task = None
                continue

            m = TASK_START_RE.match(line)
            if m:
                current_task = m.group(1)
                continue

            m = ACCURACY_RE.match(line)
            if m and current_run is not None:
                task, k, acc = m.group(1), int(m.group(2)), float(m.group(3))
                current_run["accuracies"].append((current_test_tag, task or current_task, k, acc))
                continue

    return info


def load_results_file(path):
    keys = set()
    if not path or not os.path.isfile(path):
        return keys
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            keys.add(line.split("\t", 1)[0])
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", default="final_logs",
                         help="Directory searched recursively for *.out files. Doesn't need to be the "
                              "specific job's folder -- point this at the whole final_logs/ root (or "
                              "even higher) and non-matching logs from other jobs are silently skipped. "
                              "Default: %(default)s")
    parser.add_argument("--results-file", default="results_final/KNN_ITS_family_k1.txt",
                         help="Path to the shared results file to cross-check against")
    parser.add_argument("--verbose", action="store_true",
                         help="Also print one line per log file that doesn't match this script's "
                              "header format (unrelated jobs mixed into --logs-dir). Off by default "
                              "since --logs-dir is meant to be pointed at a broad search root.")
    args = parser.parse_args()

    out_files = sorted(glob.glob(os.path.join(args.logs_dir, "**", "*.out"), recursive=True))
    if not out_files:
        print(f"No .out files found under {args.logs_dir}")
        return
    print(f"Scanning {len(out_files)} .out files under {args.logs_dir} ...\n")

    result_keys = load_results_file(args.results_file)
    if not result_keys:
        print(f"WARNING: no keys loaded from {args.results_file} (missing or empty) -- "
              f"every found accuracy will be reported as 'missing from results file'.\n")

    # grid_status[(arch, aux, repr)] = "ok" | "failed" | "ckpt_missing" | not present (never seen)
    grid_status = {}
    grid_log = {}
    recoverable = []
    no_header_count = 0

    for path in out_files:
        info = parse_log(path)

        if info["arch"] is None:
            no_header_count += 1
            if args.verbose:
                print(f"[NO HEADER]      {path}  (not an its_family_ablation_knn.sh log, or job crashed "
                      f"before printing config)")
            continue

        arch, aux = info["arch"], info["aux"]
        model_name = os.path.basename(info["ckpt"]) if info["ckpt"] else "?"

        if info["ckpt_missing"]:
            for repr_ in REPRS:
                grid_status[(arch, aux, repr_)] = "ckpt_missing"
                grid_log[(arch, aux, repr_)] = path
            print(f"[CKPT MISSING]   {path}  arch={arch} aux={aux}  ckpt={info['ckpt']}")
            continue

        for run in info["runs"]:
            repr_ = run["repr"]
            if repr_ is None:
                continue
            key_triple = (arch, aux, repr_)
            failed = repr_ in info["repr_failures"] or (info["exit_code"] not in (0, None))
            has_accuracy = len(run["accuracies"]) > 0
            if failed and not has_accuracy:
                status = "failed"
            elif not has_accuracy:
                status = "no_output"  # started (repr known) but nothing printed -- likely killed mid-run
            else:
                status = "ok"
            # Prefer "ok" if we see it from any log (in case of reruns/duplicates).
            if grid_status.get(key_triple) != "ok":
                grid_status[key_triple] = status
                grid_log[key_triple] = path

            run_name = f"knnclean_family_{arch}_{aux}_{repr_}"
            for test_tag, task, k, acc in run["accuracies"]:
                if test_tag is None or task is None:
                    continue
                rkey = f"{run_name}_{task}_{model_name}_{test_tag}_k{k}"
                if rkey not in result_keys:
                    recoverable.append((path, arch, aux, repr_, test_tag, task, k, acc, rkey))

    expected_grid = [(a, x, r) for a in ARCHES for x in AUX_TASKS for r in REPRS]
    counts = {"ok": 0, "failed": 0, "no_output": 0, "ckpt_missing": 0, "never_ran": 0}
    missing_entries = []
    for entry in expected_grid:
        status = grid_status.get(entry)
        if status is None:
            counts["never_ran"] += 1
            missing_entries.append((entry, "never_ran", None))
        else:
            counts[status] += 1
            if status != "ok":
                missing_entries.append((entry, status, grid_log[entry]))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Log files scanned:              {len(out_files)}")
    print(f"Expected grid size:             {len(expected_grid)} (2 archs x 3 aux objectives x 3 representations)")
    print(f"  OK:                           {counts['ok']}")
    print(f"  Failed (explicit error):      {counts['failed']}")
    print(f"  No output (started, no result -- likely killed mid-run): {counts['no_output']}")
    print(f"  Checkpoint missing:           {counts['ckpt_missing']}")
    print(f"  Never ran (no log at all):    {counts['never_ran']}")
    print(f"Unrelated logs skipped:         {no_header_count}  (other jobs mixed into --logs-dir; pass --verbose to list them)")

    if missing_entries:
        print(f"\nSTILL MISSING ({len(missing_entries)}) -- these are what you need to (re)run:")
        for (arch, aux, repr_), status, path in missing_entries:
            loc = f"  ({path})" if path else ""
            print(f"  arch={arch:<11} aux={aux:<7} repr={repr_:<15} status={status}{loc}")

        runnable = [e for e, s, _ in missing_entries if s != "ckpt_missing"]
        ckpt_blocked = [e for e, s, _ in missing_entries if s == "ckpt_missing"]
        if runnable:
            idx_map = {entry: i for i, entry in enumerate(expected_grid)}
            indices = sorted(idx_map[e] for e in runnable)
            print(f"\nTo resubmit only the missing-but-runnable entries ({len(indices)} tasks):")
            print(f"  sbatch --array={','.join(str(i) for i in indices)} slurm/final_scripts/its_family_ablation_knn.sh")
        if ckpt_blocked:
            print(f"\n{len(ckpt_blocked)} entries are blocked on a missing checkpoint (train it first, "
                  f"then resubmit those array indices):")
            for arch, aux, repr_ in ckpt_blocked:
                print(f"  arch={arch} aux={aux} repr={repr_}")
    else:
        print("\nEvery expected grid entry is OK.")

    if recoverable:
        print(f"\n{'=' * 70}\nRECOVERABLE: printed in log but MISSING from {args.results_file} ({len(recoverable)})")
        print("=" * 70)
        for path, arch, aux, repr_, test_tag, task, k, acc, key in recoverable:
            print(f"  {path}\n    arch={arch} aux={aux} repr={repr_} | {test_tag} {task} k={k}: "
                  f"accuracy={acc:.2f}%  (missing key: {key})")
        print("\nThese values were computed and printed to stdout but their key is not in the "
              "results file -- most likely lost to a concurrent-append race between array tasks "
              "writing to the same shared file. Recover by appending "
              "'<key>\\t<accuracy>' lines to the results file (grep the log for the exact printed "
              "line if you need the full 4-decimal precision, since knn_its_clean.py prints with "
              "2 decimals but writes with 4).")
    else:
        print(f"\nNo recoverable values found: every printed accuracy has a matching key in {args.results_file}.")


if __name__ == "__main__":
    main()