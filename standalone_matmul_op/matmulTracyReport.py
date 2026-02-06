import sys
import subprocess
import re

if len(sys.argv) != 4:
    print("Usage: python run.py <M> <K> <N>")
    sys.exit(1)

M, K, N = sys.argv[1], sys.argv[2], sys.argv[3]

MODEL_SCRIPT = "standalone_matmul_op/matmulModel.py"

# ---- Run tracy and capture output ----
proc = subprocess.run(
    [
        "python", "-m", "tracy", "-r", "-p", "-v",
        MODEL_SCRIPT, M, K, N
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=True,
)

output = proc.stdout
print(output)  # keep normal tracy logs visible

# ---- Extract CSV path from tracy log ----
match = re.search(
    r"OPs csv generated at:\s*(/.*ops_perf_results_.*\.csv)",
    output
)

if not match:
    raise RuntimeError("Could not find ops_perf_results CSV path in tracy output")

csv_path = match.group(1)

print("\n========== TT PERF REPORT ==========\n")

# ---- Run tt-perf-report on extracted path ----
subprocess.run(
    ["tt-perf-report", csv_path],
    check=True
)
