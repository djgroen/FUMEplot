"""
Regression tests confirming the decoder fixes in PR #17 are correct
against REAL external reference formats, not just our own generator's
output. Run from this directory: python3 test_real_format_compat.py

Fixture sources:
  - reordered_columns.txt, real_nist_no_velocity.txt: structure based on
    the real LAMMPS dump example in NIST's atomman docs
    (https://www.ctcms.nist.gov/potentials/atomman/tutorial/1.4.6._LAMMPS_dump_file_conversions.html),
    columns adapted to test name-based column mapping.
  - pyridine_with_comment.xyz: real pyridine coordinate data from the
    Wikipedia XYZ file format article
    (https://en.wikipedia.org/wiki/XYZ_file_format), with an added
    non-blank comment line.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brown3d_decoder import read_brown3d_file
from xyz_decoder import read_xyz_file

FIXTURES = os.path.join(os.path.dirname(__file__), "real_format_fixtures")

results = []


def check(label, fn, expect_success, filename):
    path = os.path.join(FIXTURES, filename)
    try:
        fn(path)
        ok = expect_success
        detail = "succeeded"
    except Exception as e:
        ok = not expect_success
        detail = f"raised {type(e).__name__}: {e}"
    results.append((label, ok, detail))


check(
    "LAMMPS: real dump structure, Jon's columns reordered (vx vy vz before x y z) -> should succeed",
    read_brown3d_file, True, "reordered_columns.txt",
)
check(
    "LAMMPS: real NIST dump verbatim, no velocity columns (charge/stress instead) -> should fail clearly",
    read_brown3d_file, False, "real_nist_no_velocity.txt",
)
check(
    "XYZ: real pyridine data, real non-blank comment line, element-first rows -> should fail on element parsing",
    read_xyz_file, False, "pyridine_with_comment.xyz",
)

print("=" * 80)
for label, ok, detail in results:
    print(f"[{'OK  ' if ok else 'XXXX'}] {label}\n       {detail}")
print("=" * 80)
print(f"{sum(1 for _, ok, _ in results if ok)}/{len(results)} passed")
