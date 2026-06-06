#!/bin/bash
# Integration test for multi-iteration s2v registration features.
# Requires: built TORTOISEProcess binary + DWI NIFTI data with JSON sidecar.
#
# Usage:
#   ./test_s2v_iterative.sh /path/to/TORTOISEProcess /path/to/dwi.nii
#
# The DWI data should be a multiband acquisition (MB>=2) with a .json sidecar
# containing SliceTiming. A bvec/bval pair must also be present.
#
# Each test creates an isolated temp directory and checks log output for
# expected messages. Exit code: 0 = all pass, 1 = some failed.

set -euo pipefail

TORTOISE="${1:?Usage: $0 <TORTOISEProcess_path> <dwi.nii>}"
DWI="${2:?Usage: $0 <TORTOISEProcess_path> <dwi.nii>}"
SETTINGS_DIR="$(cd "$(dirname "$0")/../../settings" && pwd)"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "TEST: $name"
    echo "=========================================="
}

check_log() {
    local logfile="$1"
    local pattern="$2"
    local desc="$3"
    if grep -q "$pattern" "$logfile" 2>/dev/null; then
        echo "  PASS: $desc"
        ((PASS++))
    else
        echo "  FAIL: $desc (pattern '$pattern' not found in log)"
        ((FAIL++))
    fi
}

check_log_absent() {
    local logfile="$1"
    local pattern="$2"
    local desc="$3"
    if grep -q "$pattern" "$logfile" 2>/dev/null; then
        echo "  FAIL: $desc (pattern '$pattern' should NOT appear in log)"
        ((FAIL++))
    else
        echo "  PASS: $desc"
        ((PASS++))
    fi
}

# ===========================================================================
# Test 1: Default settings (backward compatibility)
# ===========================================================================
run_test "Default settings (backward compatibility)"
TMPDIR1=$(mktemp -d)
$TORTOISE --up_data "$DWI" --step motioneddy --s2v 1 --temp_folder "$TMPDIR1" \
    2>&1 | tee "$TMPDIR1/log.txt" || true

LOGFILE="$TMPDIR1/log.txt"

# With defaults: s2v_smoothing_schedule=1.0,0.5,0.0, so sigma should appear
check_log "$LOGFILE" "S2V epoch" "S2V epoch logging present"
# No multistart by default
check_log_absent "$LOGFILE" "Stage 0: Large-motion" "No Stage 0 by default"
# No temporal regularization by default (s2v_lambda=0)
check_log_absent "$LOGFILE" "Applied temporal regularization" "No temporal reg by default"

# ===========================================================================
# Test 2: Convergence detection with warm-start
# ===========================================================================
run_test "Convergence detection"
TMPDIR2=$(mktemp -d)
# Create custom settings
cp "$SETTINGS_DIR/defaults.dmc" "$TMPDIR2/settings.dmc"
sed -i 's/<niter>[0-9]*</<niter>20</' "$TMPDIR2/settings.dmc"
sed -i 's/<s2v_convergence_threshold>[0-9.]*</<s2v_convergence_threshold>0.01</' "$TMPDIR2/settings.dmc"

$TORTOISE --up_data "$DWI" --step motioneddy --s2v 1 --temp_folder "$TMPDIR2" \
    2>&1 | tee "$TMPDIR2/log.txt" || true

check_log "$TMPDIR2/log.txt" "S2V convergence metric" "Convergence metric logged"
check_log "$TMPDIR2/log.txt" "S2V converged at epoch" "Early convergence triggered"

# ===========================================================================
# Test 3: Smoothing schedule + sub-iterations + temporal regularization
# ===========================================================================
run_test "Full iterative refinement (smoothing + sub-iter + temporal reg)"
TMPDIR3=$(mktemp -d)
cp "$SETTINGS_DIR/defaults.dmc" "$TMPDIR3/settings.dmc"
sed -i 's/<s2v_niter>[0-9]*</<s2v_niter>3</' "$TMPDIR3/settings.dmc"
sed -i 's/<s2v_lambda>[0-9.]*</<s2v_lambda>0.3</' "$TMPDIR3/settings.dmc"

$TORTOISE --up_data "$DWI" --step motioneddy --s2v 1 --temp_folder "$TMPDIR3" \
    2>&1 | tee "$TMPDIR3/log.txt" || true

check_log "$TMPDIR3/log.txt" "sigma=" "Smoothing sigma logged"
check_log "$TMPDIR3/log.txt" "niter=3" "Sub-iteration count logged"
check_log "$TMPDIR3/log.txt" "Applied temporal regularization" "Temporal regularization applied"

# ===========================================================================
# Test 4: Large motion correction (Stage 0)
# ===========================================================================
run_test "Large motion correction (Stage 0)"
TMPDIR4=$(mktemp -d)
cp "$SETTINGS_DIR/defaults.dmc" "$TMPDIR4/settings.dmc"
sed -i 's/<large_motion_correction>[0-9]*</<large_motion_correction>1</' "$TMPDIR4/settings.dmc"

$TORTOISE --up_data "$DWI" --step motioneddy --s2v 1 --temp_folder "$TMPDIR4" \
    2>&1 | tee "$TMPDIR4/log.txt" || true

check_log "$TMPDIR4/log.txt" "Stage 0: Large-motion" "Stage 0 executed"

# ===========================================================================
# Test 5: S2V multistart (epoch 1 only)
# ===========================================================================
run_test "S2V slice-group multistart"
TMPDIR5=$(mktemp -d)
cp "$SETTINGS_DIR/defaults.dmc" "$TMPDIR5/settings.dmc"
sed -i 's/<s2v_multistart>[0-9]*</<s2v_multistart>1</' "$TMPDIR5/settings.dmc"

$TORTOISE --up_data "$DWI" --step motioneddy --s2v 1 --temp_folder "$TMPDIR5" \
    2>&1 | tee "$TMPDIR5/log.txt" || true

# Multistart runs in epoch 1 -- just verify no crash
check_log "$TMPDIR5/log.txt" "S2V epoch 1" "Epoch 1 with multistart executed"
check_log "$TMPDIR5/log.txt" "S2V epoch 2" "Epoch 2 (standard) also executed"

# ===========================================================================
# Test 6: Progressive MAPMRI degree
# ===========================================================================
run_test "Progressive MAPMRI degree schedule"
TMPDIR6=$(mktemp -d)
cp "$SETTINGS_DIR/defaults.dmc" "$TMPDIR6/settings.dmc"
sed -i 's/<mapmri_degree_schedule>[^<]*</<mapmri_degree_schedule>2,4,4</' "$TMPDIR6/settings.dmc"

$TORTOISE --up_data "$DWI" --step motioneddy --s2v 1 --temp_folder "$TMPDIR6" \
    2>&1 | tee "$TMPDIR6/log.txt" || true

# Verify it ran without errors (MAPMRI degree changes are internal, not logged)
check_log "$TMPDIR6/log.txt" "S2V epoch" "Pipeline completed with progressive MAPMRI"

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "=========================================="
echo "RESULTS: $PASS passed, $FAIL failed"
echo "=========================================="

# Cleanup (uncomment to auto-clean)
# rm -rf "$TMPDIR1" "$TMPDIR2" "$TMPDIR3" "$TMPDIR4" "$TMPDIR5" "$TMPDIR6"

exit $((FAIL > 0 ? 1 : 0))
