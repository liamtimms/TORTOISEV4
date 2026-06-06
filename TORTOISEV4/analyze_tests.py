#!/usr/bin/env python3
"""
analyze_tests.py — Reusable analysis for tortoise_testing output.

Analyzes transform parameters, image quality, and cross-test comparisons
for any session's test results. Works on both still and motion-corrupted data.

Usage:
    python3 analyze_tests.py <output_root> [test_range] [--dti]

Options:
    --dti    Run EstimateTensor + ComputeFAMap/ComputeTRMap on each test's
             final output and compare FA/TR distributions across tests.
             Requires EstimateTensor, ComputeFAMap, ComputeTRMap in PATH
             or in the bin/ directory next to the script.

Examples:
    python3 analyze_tests.py testing_data_output/ses-still
    python3 analyze_tests.py testing_data_output/ses-still 1-6
    python3 analyze_tests.py testing_data_output/ses-3yomotion --dti
    python3 analyze_tests.py testing_data_output/ses-3yomotion 1-6 --dti
"""

import sys
import os
import json
import glob
import subprocess
import shutil
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("WARNING: nibabel not installed — image comparisons disabled")


# ─── Parameter layout from itkOkanQuadraticTransform.h ─────────────────────
# 24 params per volume. Identity: all zeros except params[6+phase]=1
# Phase encoding direction determines which scale param is 1 at identity.
PARAM_GROUPS = {
    "translation_mm":   (0, 3),     # Translation X, Y, Z
    "rotation_rad":     (3, 6),     # Rotation angles X, Y, Z
    "scale_eddy":       (6, 9),     # Scale factors (PE-dir has identity=1)
    "quadratic_cross":  (9, 12),    # Cross terms: xy, xz, yz
    "quadratic_higher": (12, 14),   # Higher quad: x²-y², 2z²-x²-y²
    "cubic":            (14, 21),   # Cubic eddy terms
    "center_offset":    (21, 24),   # Pre-transformation center offset
}


def load_transforms(path):
    """Load moteddy_transformations.txt: one line per volume, 24 params each."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().strip('[]')
            vals = [float(x) for x in line.replace(',', ' ').split()]
            rows.append(vals)
    return np.array(rows)


def detect_phase_encoding(transforms):
    """Detect PE direction from identity volume (b0).
    The b0 volume has identity transform: all zeros except params[6+phase]=1.
    """
    for i, row in enumerate(transforms):
        # Identity: params 0-5 are zero, one of 6/7/8 is 1
        if np.allclose(row[0:6], 0, atol=1e-6):
            for phase in range(3):
                if abs(row[6 + phase] - 1.0) < 1e-6:
                    others = [row[6 + j] for j in range(3) if j != phase]
                    if all(abs(v) < 1e-6 for v in others):
                        return phase, i
    return None, None


def identify_b0_volumes(transforms, phase):
    """Find b0 volume indices (identity transforms)."""
    b0s = []
    for i, row in enumerate(transforms):
        if (np.allclose(row[0:6], 0, atol=1e-6) and
                abs(row[6 + phase] - 1.0) < 1e-6):
            b0s.append(i)
    return b0s


def subtract_identity(transforms, phase):
    """Return transforms with identity subtracted so all params represent
    deviation from no-correction. Makes scale params centered at 0."""
    result = transforms.copy()
    result[:, 6 + phase] -= 1.0
    return result


def analyze_transforms(transforms, phase, label=""):
    """Print summary statistics for transform parameters."""
    nvols = transforms.shape[0]
    b0_vols = identify_b0_volumes(transforms, phase)
    dwi_mask = np.ones(nvols, dtype=bool)
    for b in b0_vols:
        dwi_mask[b] = False
    dwi_transforms = transforms[dwi_mask]

    # Subtract identity for deviation analysis
    dev = subtract_identity(dwi_transforms, phase)

    pe_labels = {0: 'X (LR)', 1: 'Y (AP)', 2: 'Z (SI)'}
    print(f"\n{'=' * 72}")
    print(f"TRANSFORM ANALYSIS: {label}")
    print(f"{'=' * 72}")
    print(f"  Volumes: {nvols} total, {len(b0_vols)} b0, {dwi_mask.sum()} DWI")
    print(f"  Phase encoding: axis {phase} ({pe_labels.get(phase, '?')})")
    print(f"  b0 volume indices: {b0_vols}")

    results = {}
    for group_name, (s, e) in PARAM_GROUPS.items():
        block = dev[:, s:e]
        abs_block = np.abs(block)

        mean_abs = np.mean(abs_block, axis=0)
        max_abs = np.max(abs_block, axis=0)
        std_vals = np.std(block, axis=0)

        results[group_name] = {
            'mean_abs': mean_abs,
            'max_abs': max_abs,
            'std': std_vals,
            'mean': np.mean(block, axis=0),
        }

        print(f"\n  {group_name} [params {s}-{e-1}]:")
        print(f"    mean_abs: {mean_abs}")
        print(f"    max_abs:  {max_abs}")
        print(f"    std:      {std_vals}")

    # Summary motion magnitude
    trans_norms = np.linalg.norm(dev[:, 0:3], axis=1)
    rot_norms = np.linalg.norm(dev[:, 3:6], axis=1)
    eddy_norms = np.linalg.norm(dev[:, 6:14], axis=1)

    print(f"\n  Motion magnitude (DWI volumes only):")
    print(f"    Translation (mm):  mean={trans_norms.mean():.4f}  "
          f"std={trans_norms.std():.4f}  max={trans_norms.max():.4f}")
    print(f"    Rotation (deg):    mean={np.degrees(rot_norms.mean()):.4f}  "
          f"std={np.degrees(rot_norms.std()):.4f}  max={np.degrees(rot_norms.max()):.4f}")
    print(f"    Eddy deviation:    mean={eddy_norms.mean():.6f}  "
          f"max={eddy_norms.max():.6f}")

    return results, dev


def compare_transforms(t1, t2, phase, label1, label2):
    """Compare transform parameters between two tests."""
    dev1 = subtract_identity(t1, phase)
    dev2 = subtract_identity(t2, phase)

    # Use minimum volume count (should be same)
    n = min(len(dev1), len(dev2))
    diff = dev2[:n] - dev1[:n]

    print(f"\n{'=' * 72}")
    print(f"TRANSFORM COMPARISON: {label1} vs {label2}")
    print(f"{'=' * 72}")

    for group_name, (s, e) in PARAM_GROUPS.items():
        block = diff[:, s:e]
        abs_block = np.abs(block)
        if np.max(abs_block) < 1e-12:
            continue  # Skip all-zero groups
        print(f"  {group_name}:")
        print(f"    mean_abs_diff: {np.mean(abs_block, axis=0)}")
        print(f"    max_abs_diff:  {np.max(abs_block, axis=0)}")

    # Overall RMSE
    rmse = np.sqrt(np.mean(diff[:, 0:14] ** 2))
    print(f"\n  Overall RMSE (params 0-13): {rmse:.8f}")


def compare_images(path1, path2, label1, label2):
    """Compare two NIfTI images (moteddy or final output)."""
    if not HAS_NIBABEL:
        return

    try:
        img1 = nib.load(path1)
        img2 = nib.load(path2)
    except Exception as e:
        print(f"  Could not load images: {e}")
        return

    d1 = img1.get_fdata()
    d2 = img2.get_fdata()

    if d1.shape != d2.shape:
        print(f"  Shape mismatch: {d1.shape} vs {d2.shape}")
        return

    # Mask out background (use threshold on mean of both)
    mean_img = (np.abs(d1) + np.abs(d2)) / 2
    mask = mean_img > np.percentile(mean_img[mean_img > 0], 5)

    diff = d1 - d2
    masked_diff = diff[mask]

    # Per-volume statistics for 4D data
    if d1.ndim == 4:
        nvols = d1.shape[3]
        vol_rmse = []
        for v in range(nvols):
            vm = mask[:, :, :, v] if mask.ndim == 4 else mask[:, :, :, 0] if mask.ndim == 4 else (mean_img[:, :, :, v] > np.percentile(mean_img[:, :, :, v][mean_img[:, :, :, v] > 0], 5) if mean_img[:, :, :, v].max() > 0 else np.ones(d1.shape[:3], dtype=bool))
            vdiff = d1[:, :, :, v] - d2[:, :, :, v]
            vol_rmse.append(np.sqrt(np.mean(vdiff[vm] ** 2)) if vm.sum() > 0 else 0)
        vol_rmse = np.array(vol_rmse)

    print(f"\n{'=' * 72}")
    print(f"IMAGE COMPARISON: {label1} vs {label2}")
    print(f"{'=' * 72}")
    print(f"  Shape: {d1.shape}")
    print(f"  Global RMSE (masked):  {np.sqrt(np.mean(masked_diff ** 2)):.6f}")
    print(f"  Global MAE (masked):   {np.mean(np.abs(masked_diff)):.6f}")
    print(f"  Max abs diff:          {np.max(np.abs(diff)):.6f}")
    print(f"  Relative RMSE:         {np.sqrt(np.mean(masked_diff ** 2)) / np.mean(np.abs(mean_img[mask])):.6f}")

    if d1.ndim == 4:
        print(f"  Per-volume RMSE: mean={vol_rmse.mean():.6f}  "
              f"std={vol_rmse.std():.6f}  max={vol_rmse.max():.6f}")


def analyze_outlier_map(path, label):
    """Analyze a native_inclusion.nii file (outlier map from repol)."""
    if not HAS_NIBABEL:
        return

    try:
        img = nib.load(path)
    except Exception:
        return

    data = img.get_fdata()
    total = data.size
    flagged = np.sum(data < 1.0)  # Inclusion < 1 means flagged
    excluded = np.sum(data == 0)

    print(f"\n{'=' * 72}")
    print(f"OUTLIER MAP: {label}")
    print(f"{'=' * 72}")
    print(f"  Shape: {data.shape}")
    print(f"  Total voxels:     {total}")
    print(f"  Flagged (< 1.0):  {flagged} ({100 * flagged / total:.2f}%)")
    print(f"  Excluded (== 0):  {excluded} ({100 * excluded / total:.2f}%)")

    if data.ndim == 4:
        nvols = data.shape[3]
        per_vol = []
        for v in range(nvols):
            vol_data = data[:, :, :, v]
            vol_flagged = np.sum(vol_data < 1.0)
            vol_total = vol_data.size
            per_vol.append(vol_flagged / vol_total)
        per_vol = np.array(per_vol)
        print(f"  Per-volume flagged fraction: mean={per_vol.mean():.4f}  "
              f"max={per_vol.max():.4f}  vols>5%={np.sum(per_vol > 0.05)}")


def analyze_s2v_transforms(path, label):
    """Analyze s2v_transformations.txt — per-slice transforms."""
    if not os.path.exists(path):
        return

    with open(path) as f:
        lines = f.readlines()

    print(f"\n{'=' * 72}")
    print(f"S2V TRANSFORMS: {label}")
    print(f"{'=' * 72}")
    print(f"  Lines (slices*volumes): {len(lines)}")

    # Parse all transforms
    all_params = []
    for line in lines:
        line = line.strip().strip('[]')
        if not line:
            continue
        vals = [float(x) for x in line.replace(',', ' ').split()]
        all_params.append(vals)

    if not all_params:
        print("  No transform data found")
        return

    params = np.array(all_params)
    print(f"  Params per slice: {params.shape[1] if params.ndim > 1 else 'N/A'}")

    # S2V uses 6 rigid params (translation + rotation)
    if params.shape[1] >= 6:
        trans = params[:, 0:3]
        rot = params[:, 3:6]
        trans_norms = np.linalg.norm(trans, axis=1)
        rot_norms = np.linalg.norm(rot, axis=1)

        print(f"  Translation (mm):  mean={trans_norms.mean():.6f}  "
              f"std={trans_norms.std():.6f}  max={trans_norms.max():.6f}")
        print(f"  Rotation (deg):    mean={np.degrees(rot_norms.mean()):.6f}  "
              f"std={np.degrees(rot_norms.std()):.6f}  max={np.degrees(rot_norms.max()):.6f}")

        # Flag if s2v corrections are large (suspicious on still data)
        if trans_norms.max() > 2.0:
            print(f"  *** WARNING: Large S2V translation detected (>{trans_norms.max():.2f} mm)")
        if np.degrees(rot_norms.max()) > 2.0:
            print(f"  *** WARNING: Large S2V rotation detected (>{np.degrees(rot_norms.max()):.2f} deg)")


def find_binaries():
    """Find EstimateTensor, ComputeFAMap, ComputeTRMap binaries."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(script_dir, 'bin'),
        os.path.join(script_dir, '..', 'bin'),
    ]

    binaries = {}
    for name in ['EstimateTensor', 'ComputeFAMap', 'ComputeTRMap']:
        path = shutil.which(name)
        if path:
            binaries[name] = path
            continue
        for d in search_dirs:
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                binaries[name] = os.path.realpath(candidate)
                break
        if name not in binaries:
            return None
    return binaries


def run_dti_fitting(final_nii, binaries):
    """Run EstimateTensor + ComputeFAMap + ComputeTRMap on a final output.
    Returns paths to (FA, TR, A0) nii files, or (None, None, None) on failure."""
    base = final_nii.replace('.nii.gz', '').replace('.nii', '')
    dt_file = f"{base}_L1_DT.nii"
    fa_file = f"{base}_L1_DT_FA.nii"
    tr_file = f"{base}_L1_DT_TR.nii"
    a0_file = f"{base}_L1_AM.nii"

    # Skip if already computed
    if os.path.exists(fa_file) and os.path.exists(tr_file):
        return fa_file, tr_file, a0_file if os.path.exists(a0_file) else None

    try:
        if not os.path.exists(dt_file):
            subprocess.run(
                [binaries['EstimateTensor'], '-i', final_nii],
                capture_output=True, timeout=600
            )
        if not os.path.exists(dt_file):
            return None, None, None

        if not os.path.exists(fa_file):
            subprocess.run(
                [binaries['ComputeFAMap'], dt_file],
                capture_output=True, timeout=120
            )

        if not os.path.exists(tr_file):
            subprocess.run(
                [binaries['ComputeTRMap'], dt_file],
                capture_output=True, timeout=120
            )

        fa_out = fa_file if os.path.exists(fa_file) else None
        tr_out = tr_file if os.path.exists(tr_file) else None
        a0_out = a0_file if os.path.exists(a0_file) else None
        return fa_out, tr_out, a0_out

    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  DTI fitting failed: {e}")
        return None, None, None


def make_brain_mask(a0_path):
    """Create brain mask from A0 (b=0 signal amplitude) image only.
    No diffusion metrics used — avoids circular bias."""
    if not HAS_NIBABEL or not a0_path or not os.path.exists(a0_path):
        return None
    a0_data = nib.load(a0_path).get_fdata()
    # Otsu-like: threshold at 15th percentile of nonzero voxels
    nonzero = a0_data[a0_data > 0]
    if nonzero.size == 0:
        return None
    threshold = np.percentile(nonzero, 15)
    return a0_data > threshold


def analyze_fa_tr(fa_path, tr_path, brain_mask, label):
    """Analyze FA and TR (=3*MD) distributions over an A0-derived brain mask.
    No WM segmentation — reports whole-brain statistics only."""
    if not HAS_NIBABEL:
        return None

    stats = {}

    if fa_path and os.path.exists(fa_path):
        fa_data = nib.load(fa_path).get_fdata()
        if brain_mask is not None:
            fa_brain = fa_data[brain_mask & (fa_data >= 0) & (fa_data <= 1)]
        else:
            fa_brain = fa_data[(fa_data > 0.01) & (fa_data <= 1)]

        if fa_brain.size > 0:
            stats['fa_mean'] = float(np.mean(fa_brain))
            stats['fa_std'] = float(np.std(fa_brain))
            stats['fa_median'] = float(np.median(fa_brain))
            stats['fa_p10'] = float(np.percentile(fa_brain, 10))
            stats['fa_p25'] = float(np.percentile(fa_brain, 25))
            stats['fa_p75'] = float(np.percentile(fa_brain, 75))
            stats['fa_p90'] = float(np.percentile(fa_brain, 90))
            stats['brain_voxels'] = int(fa_brain.size)

    if tr_path and os.path.exists(tr_path):
        tr_data = nib.load(tr_path).get_fdata()
        if brain_mask is not None:
            tr_brain = tr_data[brain_mask & (tr_data > 0)]
        else:
            tr_brain = tr_data[tr_data > 0]

        if tr_brain.size > 0:
            md_brain = tr_brain / 3.0
            stats['md_mean'] = float(np.mean(md_brain))
            stats['md_std'] = float(np.std(md_brain))
            stats['md_median'] = float(np.median(md_brain))

    if not stats:
        return None

    print(f"\n{'=' * 72}")
    print(f"DTI SCALARS: {label}")
    print(f"{'=' * 72}")
    if 'fa_mean' in stats:
        print(f"  Brain voxels: {stats['brain_voxels']}")
        print(f"  FA:  mean={stats['fa_mean']:.4f}  std={stats['fa_std']:.4f}  "
              f"median={stats['fa_median']:.4f}")
        print(f"       [p10={stats['fa_p10']:.4f}  p25={stats['fa_p25']:.4f}  "
              f"p75={stats['fa_p75']:.4f}  p90={stats['fa_p90']:.4f}]")
    if 'md_mean' in stats:
        print(f"  MD:  mean={stats['md_mean']:.1f}  std={stats['md_std']:.1f}  "
              f"median={stats['md_median']:.1f} um^2/s")

    return stats


def compare_dti_stats(stats1, stats2, label1, label2):
    """Compare DTI scalar statistics between two tests."""
    if stats1 is None or stats2 is None:
        return

    print(f"\n  {label1} -> {label2}:")
    for key, desc, fmt in [
        ('fa_mean', 'FA mean', '.4f'),
        ('fa_median', 'FA median', '.4f'),
        ('fa_p10', 'FA p10', '.4f'),
        ('fa_p90', 'FA p90', '.4f'),
        ('md_mean', 'MD mean', '.1f'),
    ]:
        v1 = stats1.get(key)
        v2 = stats2.get(key)
        if v1 is None or v2 is None:
            continue
        diff = v2 - v1
        pct = 100 * diff / v1 if v1 != 0 else 0
        print(f"    {desc}: {v1:{fmt}} -> {v2:{fmt}} ({diff:+{fmt}}, {pct:+.2f}%)")


def find_test_dirs(output_root):
    """Find all completed test directories and their working dirs."""
    tests = {}
    for d in sorted(glob.glob(os.path.join(output_root, "working_test_*"))):
        name = os.path.basename(d).replace("working_", "")
        # Extract test number
        parts = name.split('_')
        try:
            num = int(parts[1])
        except (IndexError, ValueError):
            continue

        working_dir = d
        output_dir = os.path.join(output_root, name)

        # Find key files
        moteddy_trans = glob.glob(os.path.join(working_dir, "*_moteddy_transformations.txt"))
        moteddy_nii = glob.glob(os.path.join(working_dir, "*_moteddy.nii"))
        s2v_trans = glob.glob(os.path.join(working_dir, "*_s2v_transformations.txt"))
        inclusion = glob.glob(os.path.join(working_dir, "*_native_inclusion.nii"))
        final_nii = glob.glob(os.path.join(output_dir, "*.nii"))

        tests[num] = {
            'name': name,
            'working_dir': working_dir,
            'output_dir': output_dir,
            'moteddy_trans': moteddy_trans[0] if moteddy_trans else None,
            'moteddy_nii': moteddy_nii[0] if moteddy_nii else None,
            's2v_trans': s2v_trans[0] if s2v_trans else None,
            'inclusion': inclusion[0] if inclusion else None,
            'final_nii': final_nii[0] if final_nii else None,
        }

    return tests


def parse_test_range(range_str, max_test):
    """Parse test range like '1-3', '4', '1-3,8-10', 'all'."""
    if range_str == 'all':
        return list(range(1, max_test + 1))

    result = []
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    do_dti = '--dti' in flags

    output_root = args[0] if args else None
    range_str = args[1] if len(args) > 1 else 'all'

    if not output_root:
        print(__doc__)
        sys.exit(0)

    if not os.path.isdir(output_root):
        print(f"Error: {output_root} is not a directory")
        sys.exit(1)

    # Find all completed tests
    all_tests = find_test_dirs(output_root)
    if not all_tests:
        print(f"No test results found in {output_root}")
        sys.exit(1)

    max_test = max(all_tests.keys())
    requested = parse_test_range(range_str, max_test)
    tests = {k: v for k, v in all_tests.items() if k in requested}

    print(f"Output root: {output_root}")
    print(f"Found tests: {sorted(all_tests.keys())}")
    print(f"Analyzing:   {sorted(tests.keys())}")

    # Read timing data
    timing_file = os.path.join(output_root, "timing.tsv")
    timing = {}
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] != 'test_name':
                    timing[parts[0]] = {
                        'exit_code': int(parts[1]),
                        'elapsed': int(parts[2])
                    }

    # ─── Timing Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("TIMING SUMMARY")
    print(f"{'=' * 72}")
    for num in sorted(tests.keys()):
        t = tests[num]
        tinfo = timing.get(t['name'], {})
        elapsed = tinfo.get('elapsed', 0)
        exit_code = tinfo.get('exit_code', '?')
        mins, secs = divmod(elapsed, 60)
        status = "OK" if exit_code == 0 else f"FAIL({exit_code})"
        files = []
        if t['moteddy_trans']:
            files.append('trans')
        if t['s2v_trans']:
            files.append('s2v')
        if t['inclusion']:
            files.append('outlier')
        if t['final_nii']:
            files.append('final')
        print(f"  {num:02d} {t['name']:42s}  {status:8s}  "
              f"{mins:2d}m{secs:02d}s  [{', '.join(files)}]")

    # ─── Detect phase encoding from first available test ──────────────
    phase = None
    for num in sorted(tests.keys()):
        if tests[num]['moteddy_trans']:
            t = load_transforms(tests[num]['moteddy_trans'])
            phase, b0_idx = detect_phase_encoding(t)
            if phase is not None:
                pe_labels = {0: 'X (LR)', 1: 'Y (AP)', 2: 'Z (SI)'}
                print(f"\nPhase encoding: axis {phase} ({pe_labels.get(phase, '?')}), "
                      f"b0 volume: {b0_idx}")
                break

    if phase is None:
        print("\nWARNING: Could not detect phase encoding direction")
        phase = 1  # Default to AP

    # ─── Per-test transform analysis ──────────────────────────────────
    loaded_transforms = {}
    for num in sorted(tests.keys()):
        t = tests[num]
        if t['moteddy_trans']:
            transforms = load_transforms(t['moteddy_trans'])
            loaded_transforms[num] = transforms
            analyze_transforms(transforms, phase, f"Test {num:02d}: {t['name']}")

        if t['s2v_trans']:
            analyze_s2v_transforms(t['s2v_trans'], f"Test {num:02d}: {t['name']}")

        if t['inclusion']:
            analyze_outlier_map(t['inclusion'], f"Test {num:02d}: {t['name']}")

    # ─── Cross-test comparisons ───────────────────────────────────────
    # Define comparison pairs based on test battery structure
    comparison_pairs = [
        # Phase 1: rigid vs quadratic
        (1, 2, "Rigid vs Quadratic baseline"),
        # Phase 2: incremental features
        (2, 3, "Baseline vs Iterative (niter=4)"),
        (3, 4, "Iterative vs +repol"),
        (3, 5, "Iterative vs +s2v"),
        (5, 6, "S2V vs S2V+repol"),
        # Phase 3a: isolated knobs vs baseline
        (6, 7, "Baseline vs +smoothing (isolated)"),
        (6, 8, "Baseline vs +subiters (isolated)"),
        (6, 9, "Baseline vs +warm_start (isolated)"),
        (6, 10, "Baseline vs +temporal_reg (isolated)"),
        # Phase 3b: cumulative
        (6, 14, "Baseline vs fully-tuned"),
        # Phase 4: motion flags
        (6, 16, "Baseline vs +LMC"),
        (6, 17, "Baseline vs +s2v_multistart"),
        (6, 18, "Baseline vs +LMC+multistart"),
        (14, 19, "Tuned vs +LMC"),
        (14, 20, "Tuned vs +s2v_multistart"),
        (14, 21, "Tuned vs +LMC+multistart"),
    ]

    print(f"\n\n{'#' * 72}")
    print("CROSS-TEST TRANSFORM COMPARISONS")
    print(f"{'#' * 72}")

    for t1, t2, desc in comparison_pairs:
        if t1 in loaded_transforms and t2 in loaded_transforms:
            compare_transforms(
                loaded_transforms[t1], loaded_transforms[t2],
                phase, f"Test {t1:02d}", f"Test {t2:02d} ({desc})"
            )

    # ─── Image comparisons ────────────────────────────────────────────
    if HAS_NIBABEL:
        image_pairs = [
            (1, 2, "Rigid vs Quadratic"),
            (2, 3, "Baseline vs Iterative"),
            (6, 14, "Baseline vs Fully-tuned"),
            (6, 16, "Baseline vs +LMC"),
            (6, 17, "Baseline vs +multistart"),
            (14, 19, "Tuned vs +LMC"),
            (14, 20, "Tuned vs +multistart"),
        ]

        print(f"\n\n{'#' * 72}")
        print("IMAGE COMPARISONS (moteddy output)")
        print(f"{'#' * 72}")

        for t1, t2, desc in image_pairs:
            if (t1 in tests and t2 in tests and
                    tests[t1]['moteddy_nii'] and tests[t2]['moteddy_nii']):
                compare_images(
                    tests[t1]['moteddy_nii'], tests[t2]['moteddy_nii'],
                    f"Test {t1:02d}", f"Test {t2:02d} ({desc})"
                )

    # ─── DTI scalar analysis (optional) ──────────────────────────────
    dti_stats = {}
    if do_dti:
        binaries = find_binaries()
        if binaries is None:
            print("\nWARNING: --dti requested but EstimateTensor/ComputeFAMap/ComputeTRMap "
                  "not found. Skipping DTI analysis.")
        elif not HAS_NIBABEL:
            print("\nWARNING: --dti requested but nibabel not installed. Skipping.")
        else:
            print(f"\n\n{'#' * 72}")
            print("DTI SCALAR ANALYSIS (FA / MD)")
            print(f"{'#' * 72}")
            print(f"  Binaries: {', '.join(f'{k}={v}' for k, v in binaries.items())}")

            # Fit DTI and compute scalars for each test
            for num in sorted(tests.keys()):
                t = tests[num]
                if not t['final_nii']:
                    continue
                # Filter out structural_0.nii
                final = t['final_nii']
                if 'structural' in os.path.basename(final):
                    # Find the actual DWI output
                    candidates = [f for f in glob.glob(os.path.join(t['output_dir'], '*.nii'))
                                  if 'structural' not in os.path.basename(f)]
                    if candidates:
                        final = candidates[0]
                    else:
                        continue

                print(f"\n  Fitting test {num:02d}...")
                fa_path, tr_path, a0_path = run_dti_fitting(final, binaries)
                if fa_path is None:
                    print(f"    Skipped (fitting failed)")
                    continue

                brain_mask = make_brain_mask(a0_path)
                stats = analyze_fa_tr(fa_path, tr_path, brain_mask,
                                      f"Test {num:02d}: {t['name']}")
                if stats:
                    dti_stats[num] = stats

            # Cross-test DTI comparisons
            if dti_stats:
                print(f"\n{'=' * 72}")
                print("DTI SCALAR COMPARISONS")
                print(f"{'=' * 72}")

                dti_pairs = [
                    (1, 2, "Rigid vs Quadratic"),
                    (2, 3, "Baseline vs Iterative"),
                    (2, 6, "No-S2V vs S2V+repol"),
                    (6, 14, "Baseline vs Fully-tuned"),
                    (6, 16, "Baseline vs +LMC"),
                    (6, 17, "Baseline vs +multistart"),
                    (14, 19, "Tuned vs +LMC"),
                    (14, 20, "Tuned vs +multistart"),
                ]

                for t1, t2, desc in dti_pairs:
                    if t1 in dti_stats and t2 in dti_stats:
                        compare_dti_stats(dti_stats[t1], dti_stats[t2],
                                          f"Test {t1:02d}", f"Test {t2:02d} ({desc})")

    # ─── Diagnostic flags ─────────────────────────────────────────────
    print(f"\n\n{'#' * 72}")
    print("DIAGNOSTIC FLAGS")
    print(f"{'#' * 72}")

    flags_raised = []

    for num in sorted(tests.keys()):
        if num not in loaded_transforms:
            continue
        t = loaded_transforms[num]
        dev = subtract_identity(t, phase)

        # Check for unexpectedly large motion on still data
        trans_norms = np.linalg.norm(dev[:, 0:3], axis=1)
        rot_norms = np.linalg.norm(dev[:, 3:6], axis=1)

        if trans_norms.max() > 3.0:
            msg = f"Test {num:02d}: Large translation ({trans_norms.max():.2f} mm)"
            flags_raised.append(msg)

        if np.degrees(rot_norms.max()) > 3.0:
            msg = f"Test {num:02d}: Large rotation ({np.degrees(rot_norms.max()):.2f} deg)"
            flags_raised.append(msg)

        # Check for eddy params diverging from identity
        eddy_norms = np.linalg.norm(dev[:, 6:14], axis=1)
        if eddy_norms.max() > 0.1:
            msg = f"Test {num:02d}: Large eddy deviation ({eddy_norms.max():.4f})"
            flags_raised.append(msg)

    # Check Phase 4 motion flags had minimal effect on still data
    for t1, t2, desc in [(6, 16, "LMC"), (6, 17, "multistart"), (14, 19, "LMC tuned"), (14, 20, "multistart tuned")]:
        if t1 in loaded_transforms and t2 in loaded_transforms:
            diff = subtract_identity(loaded_transforms[t2], phase) - subtract_identity(loaded_transforms[t1], phase)
            rmse = np.sqrt(np.mean(diff[:, 0:14] ** 2))
            if rmse > 0.01:
                msg = f"Tests {t1:02d} vs {t2:02d}: {desc} changed transforms (RMSE={rmse:.6f})"
                flags_raised.append(msg)

    if flags_raised:
        print("\n  Flags raised:")
        for f in flags_raised:
            print(f"    *** {f}")
    else:
        print("\n  No diagnostic flags raised.")

    print(f"\n{'=' * 72}")
    print("Analysis complete.")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
