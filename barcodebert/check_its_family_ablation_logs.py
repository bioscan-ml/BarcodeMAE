"""
Audit tool for its_family_ablation_knn.sh's 6-task SLURM array.

Each of the 6 array tasks (arch in {maelm, transformer} x aux in
{binary, triplet, ce}) loops over all 3 representations (tokens, cls,
tokens_with_cls) SEQUENTIALLY within one job, running knn_its_clean.py three
times and appending every result to the single shared file
results_final/KNN_ITS_family_k1.txt. Two things can cause a result to be
"missing" even though the run happened:

  1. The job/task genuinely failed or was killed partway through (OOM,
     timeout, crash) -- some or all of its 3 representations never ran.
  2. The value WAS computed and printed to stdout, but its results-file
     line got lost: knn_its_clean.py writes each (task,k) result to the
     shared file in the same loop iteration it's printed in, but if this
     array task's write raced with another concurrently-running array
     task's append to that same file (parallel filesystem, no locking),
     the line can be silently dropped even though the run succeeded.

This script tells the two cases apart: it parses every log for (a) which of
the 6 (arch,aux) tasks completed vs failed, (b) which representations
finished within each task, and (c) cross-checks every accuracy value
printed to stdout against the actual results file, flagging any that are
printed in the log but absent from the file (case 2 -- recoverable without
rerunning anything).

Run on the cluster (standard library only):

    python barcodebert/check_its_family_ablation_logs.py \\
        --logs-dir final_logs/<ARRAY_JOB_ID> \\
        --results-file results_final/KNN_ITS_family_k1.txt

--logs-dir accepts either one job's log directory (containing
<ARRAY_JOB_ID>_<TASK_ID>.out files) or a parent directory containing several
such job directories (e.g. if the array was resubmitted after partial
failures) -- it is searched recursively for *.out files.
"""
import argparse
import glob
import os
import re

# Mirrors its_family_ablation_knn.sh's grid (ARCHS/AUX_TASKS arrays).
ARCHES = ["maelm", "transformer"]
AUX_TASKS = ["binary", "triplet", "ce"]
REPRS = ["tokens", "cls", "tokens_with_cls"]
TEST_SET_TAGS = {
    "Test1 (Yeast)": "test1",
    "Test2 (Filamentous)": "test2",
    "Test3 (MycoAI)": "test3",
}

HEADER_RE = re.compile(r"^Arch: (?P<arch>\S+) \| Aux: (?P<aux>\S+) \| Ckpt: (?P<ckpt>\S+)")
CKPT_MISSING_RE = re.compile(r"^ERROR: checkpoint not found at (\S+)")
REPR_FAILED_RE = re.compile(r"^ERROR: knn_its_clean\.py failed for (\S+)")
ALL_DONE_RE = re.compile(r"^All done at: .* \| exit: (\d+)")
RUN_NAME_IN_NAMESPACE_RE = re.compile(r"run_name='([^']*)'")
TESTSET_START_RE = re.compile(r"^(Test\d \([^)]+\)): (\d+) query specimens")
TASK_START_RE = re.compile(r"^\s*--- (\S+) ---")
ACCURACY_RE = re.compile(r"^\s*\[(\S+)\] k=(\d+): accuracy=([\d.]+)%")


def parse_log(path):
    """Returns a dict describing one array task's log (arch, aux, and,
    for each of the up to 3 sequential representation runs within it, its
    repr name, whether it failed, and every (test_tag, task, k, accuracy)
    it printed)."""
    info = {
        "path": path,
        "header": None,
        "ckpt_missing": False,
        "exit_code": None,
        "repr_failures": set(),   # reprs explicitly reported as failed
        "runs": [],  # list of {"repr": str|None, "accuracies": [...]}
    }
    current_run = None
    current_test_tag = None
    current_task = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = HEADER_RE.match(line)
            if m:
                info["header"] = m.groupdict()
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
                current_run = {"repr": repr_, "run_name": run_name, "accuracies": []}
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

    expected_grid = {(arch, aux) for arch in ARCHES for aux in AUX_TASKS}
    seen_grid = set()
    recoverable = []
    no_header_count = 0
    ckpt_missing_count = 0
    fully_ok_count = 0
    partial_count = 0

    for path in out_files:
        info = parse_log(path)

        if info["header"] is None:
            no_header_count += 1
            if args.verbose:
                print(f"[NO HEADER]      {path}  (not an its_family_ablation_knn.sh log, or job crashed "
                      f"before printing config)")
            continue

        h = info["header"]
        arch, aux = h["arch"], h["aux"]
        seen_grid.add((arch, aux))
        model_name = os.path.basename(h["ckpt"])

        if info["ckpt_missing"]:
            ckpt_missing_count += 1
            print(f"[CKPT MISSING]   {path}  arch={arch} aux={aux}  ckpt={h['ckpt']}")
            continue

        ran_reprs = {r["repr"] for r in info["runs"] if r["repr"] is not None}
        missing_reprs = set(REPRS) - ran_reprs
        failed_reprs = info["repr_failures"]

        status_bits = []
        if missing_reprs:
            status_bits.append(f"never ran: {sorted(missing_reprs)}")
        if failed_reprs:
            status_bits.append(f"reported failed: {sorted(failed_reprs)}")
        if info["exit_code"] not in (0, None) and not status_bits:
            status_bits.append(f"nonzero exit {info['exit_code']}, but all 3 reprs ran and printed results")
        if info["exit_code"] is None and not missing_reprs:
            status_bits.append("no 'All done' line (job likely killed/timed out at the very end)")

        if status_bits:
            partial_count += 1
            print(f"[PARTIAL]        {path}  arch={arch} aux={aux}  " + "; ".join(status_bits))
        else:
            fully_ok_count += 1

        # Cross-check every printed accuracy against the results file, for
        # every representation run found in this log (even failed ones may
        # have printed some accuracies before failing).
        for run in info["runs"]:
            repr_ = run["repr"]
            if repr_ is None:
                continue
            run_name = f"knnclean_family_{arch}_{aux}_{repr_}"
            for test_tag, task, k, acc in run["accuracies"]:
                if test_tag is None or task is None:
                    continue
                key = f"{run_name}_{task}_{model_name}_{test_tag}_k{k}"
                if key not in result_keys:
                    recoverable.append((path, arch, aux, repr_, test_tag, task, k, acc, key))

    missing_grid = sorted(expected_grid - seen_grid)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Log files scanned:              {len(out_files)}")
    print(f"Expected (arch,aux) tasks:      {len(expected_grid)} (2 archs x 3 aux objectives)")
    print(f"Tasks with a log:               {len(seen_grid)}")
    print(f"  Fully OK (all 3 reprs ran):   {fully_ok_count}")
    print(f"  Partial / problem:            {partial_count}")
    print(f"  Checkpoint missing:           {ckpt_missing_count}")
    print(f"Unrelated logs skipped:          {no_header_count}  (other jobs mixed into --logs-dir; pass --verbose to list them)")

    if missing_grid:
        print(f"\n(arch,aux) tasks with NO log file at all ({len(missing_grid)}):")
        for arch, aux in missing_grid:
            print(f"  arch={arch} aux={aux}")
    else:
        print("\nEvery expected (arch,aux) task has at least one log file.")

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