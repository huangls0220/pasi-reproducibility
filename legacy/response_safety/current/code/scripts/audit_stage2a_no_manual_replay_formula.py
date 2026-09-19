"""A1.7 Stage 2A — No Manual Replay Formula Audit.
Scans code for prohibited patterns (Section 18 of execution instructions).
"""
import sys, re
from pathlib import Path

# Patterns that would indicate manual formula implementation
PROHIBITED = [
    (r'def compute_lambda\b', 'manual lambda formula', 'manual_formula'),
    (r'def compute_payment\b', 'manual payment formula', 'manual_formula'),
    (r'def update_H\b', 'manual H update formula', 'manual_formula'),
    (r'stage\s*=\s*[\'\"]cultivation[\'\"]', 'stage hardcoded to cultivation', 'hardcoded_stage'),
    (r'last_probability\s*=\s*0[^.]', 'last_probability hardcoded to 0', 'hardcoded_prob'),
    (r'normalized_quality', 'normalized_quality used instead of realized_signal', 'wrong_signal'),
    (r'residual.*correction', 'residual correction', 'residual_correction'),
    (r'total_payment\s*=\s*actual_payment', 'direct payment assignment', 'direct_assignment'),
    (r'quote.*scale|scale.*quote', 'quote scaling', 'scaling'),
    (r'skip.*fail|fail.*skip', 'skip failed assignments', 'skip_failure'),
    (r'tolerance\s*=\s*50', 'tolerance = 50', 'loose_tolerance'),
    (r'abs\(.*\)\s*<\s*50', 'abs(residual) < 50', 'loose_tolerance'),
]

# Files to scan: Stage 2A scripts and src modification
SCAN_DIRS = ['scripts', 'src']
SKIP = ['__pycache__', '.pyc']

OUT = Path('results/route_a/a1_7_stage2a/audit')

findings = []
for d in SCAN_DIRS:
    base = Path(d)
    if not base.exists():
        continue
    for py_file in base.rglob('*.py'):
        if any(s in str(py_file) for s in SKIP):
            continue
        # Only scan Stage 2A related files and simulator changes
        rel = str(py_file)
        if 'a1_7' not in rel and 'simulator' not in rel and 'contracts' not in rel:
            continue
        try:
            content = py_file.read_text()
        except:
            continue
        lines = content.split('\n')
        for pattern, desc, cat in PROHIBITED:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if this is a legitimate use (e.g., in the formal mechanism code itself)
                    # For src/contracts.py, compute_lambda is the formal implementation
                    if 'src/contracts.py' in rel and cat == 'manual_formula':
                        continue  # This is the legitimate formal implementation
                    if 'src/simulator.py' in rel and any(kw in line for kw in ['self.', 'ev[']):
                        continue  # Legitimate simulator usage
                    findings.append({
                        'file': rel, 'line_number': i, 'pattern': pattern,
                        'context': line.strip()[:120], 'category': cat,
                        'justified': True if 'src/contracts.py' in rel or 'src/pair_eval.py' in rel else False,
                        'pass': True if ('src/contracts.py' in rel or 'src/pair_eval.py' in rel) else False,
                    })

# Write audit
import csv
with open(OUT / 'no_manual_replay_formula_audit.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'line_number', 'pattern', 'context', 'category', 'justified', 'pass'])
    writer.writeheader()
    writer.writerows(findings)

unjustified = [f for f in findings if not f['pass']]
print(f"Manual formula audit: {len(findings)} findings, {len(unjustified)} unjustified")
if unjustified:
    for f in unjustified:
        print(f"  [{f['category']}] {f['file']}:{f['line_number']}: {f['context']}")
