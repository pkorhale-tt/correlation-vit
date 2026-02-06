import subprocess
import re
from openpyxl import Workbook

SCRIPT = "standalone_matmul_op/matmulTracyReport.py"

START = 32
END = 2000
STEP = 32

wb = Workbook()
ws = wb.active
ws.title = "Matmul_Report"

# Excel header (exactly your table)
ws.append([
    "M", "N", "K",
    "ID", "Total %", "Bound", "OP Code", "Device",
    "Device Time (us)", "Op-to-Op Gap (us)",
    "Cores", "DRAM", "DRAM %",
    "FLOPs", "FLOPs %",
    "Math Fidelity"
])

for size in range(START, END + 1, STEP):
    M = N = K = size
    print(f"Running {M}x{N}x{K}")

    proc = subprocess.run(
        ["python", SCRIPT, str(M), str(N), str(K)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # hide noise
        text=True,
    )

    output = proc.stdout

    # Pick only Matmul rows
    matmul_rows = [
        line for line in output.splitlines()
        if "MatmulDeviceOperation" in line
    ]

    if len(matmul_rows) < 3:
        print(f"Skipping {size}: insufficient matmul rows")
        continue

    row = matmul_rows[2]  # 3rd matmul only

    # Robust split (table uses multi-space separation)
    cols = re.split(r"\s{2,}", row.strip())

    if len(cols) < 13:
        print(f"Skipping {size}: parse error")
        continue

    ws.append([
        M, N, K,
        cols[0],    # ID
        cols[1],    # Total %
        cols[2],    # Bound
        cols[3],    # OP Code
        cols[4],    # Device
        cols[5].replace("μs", ""),
        cols[6].replace("μs", "") if "μs" in cols[6] else "",
        cols[7],    # Cores
        cols[8],    # DRAM
        cols[9],    # DRAM %
        cols[10],   # FLOPs
        cols[11],   # FLOPs %
        cols[12],   # Math Fidelity
    ])

# Save Excel
output_file = "matmul_fidelity_sweep.xlsx"
wb.save(output_file)

print(f"\n✅ Excel generated: {output_file}")

