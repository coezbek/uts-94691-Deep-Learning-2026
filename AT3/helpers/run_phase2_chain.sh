#!/usr/bin/env bash
# Chain-run the new Phase 2 experiments. Each is logged to /tmp/eXX.log.
set -u
cd "$(dirname "$0")"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
PY=/home/coezbek/dev/2026/AT3_training/.venv/bin/python

run() {
    local nb="$1"; local tag="$2"; local log="$3"
    echo "[chain] === $tag === $(date -Iseconds)"
    "$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=14400 "$nb" > "$log" 2>&1
    rc=$?
    echo "[chain] $tag exit=$rc"
    return $rc
}

# 1. SCST with corpus-IDF on E05 (~10 min)
run phase2_e10_scst_corpus_idf.py E10 /tmp/e10.log || echo "[chain] WARN: E10 failed, continuing"

# 2. E05-anchored ensemble (inference only, ~5 min)
run phase2_e11_ensemble_e05.py E11 /tmp/e11.log || echo "[chain] WARN: E11 failed, continuing"

# 3. Small regularised decoder on CLIP features (~10 min)
run phase2_e12_small_decoder.py E12 /tmp/e12.log || echo "[chain] WARN: E12 failed, continuing"

# 4. SigLIP2 encoder swap (~20 min: extract + train)
run phase2_e13_siglip2.py E13 /tmp/e13.log || echo "[chain] WARN: E13 failed, continuing"

# 5. MBR decoding (~10 min)
run phase2_e14_mbr.py E14 /tmp/e14.log || echo "[chain] WARN: E14 failed, continuing"

echo "[chain] all done $(date -Iseconds)"
"$PY" build_summary_report.py
