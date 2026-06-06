#!/usr/bin/env python3
"""
dice_analysis.py — Per-volume DICE scoring for ses-largermoving tests.

For each test, loads the corrected NIfTI and the test's own structural_0.nii
(b0 reference). Otsu-thresholds each volume to get a binary brain mask and
computes DICE against the Otsu-thresholded structural reference.

A DICE drop to ~0.7 is severe for a brain-shaped object because the center
still overlaps even when badly rotated — so 0.7 means major misalignment.

Usage:
    python3 dice_analysis.py <remote_output_dir> <local_output_dir> [--vol N]

    remote_output_dir  testing_data_output/ses-largermoving (tests 1-18)
    local_output_dir   testing_data_output/ses-largermoving-local (test 24)
    --vol N            also print per-volume detail for volume N (default: 40)
"""

import sys
import os
import glob
import numpy as np
import nibabel as nib


# ─── Otsu thresholding (no skimage needed) ──────────────────────────────────

def otsu_threshold(data):
    """Compute Otsu threshold from a flattened array of positive values."""
    if len(data) == 0:
        return 0.0
    counts, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return 0.0
    weight1 = np.cumsum(counts)
    weight2 = total - weight1
    mean1 = np.cumsum(counts * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((counts * bin_centers)[::-1])[::-1]) / np.maximum(weight2, 1)
    variance = weight1 * weight2 * (mean1 - mean2) ** 2
    idx = np.argmax(variance)
    return bin_centers[idx]


def brain_mask(vol_3d):
    """Return binary brain mask from a 3D volume using Otsu thresholding."""
    flat = vol_3d.ravel()
    pos = flat[flat > 0]
    if len(pos) == 0:
        return np.zeros(vol_3d.shape, dtype=bool)
    thresh = otsu_threshold(pos)
    return vol_3d > thresh


def dice(mask_a, mask_b):
    """Compute DICE coefficient between two binary masks."""
    inter = np.sum(mask_a & mask_b)
    denom = np.sum(mask_a) + np.sum(mask_b)
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


# ─── Test discovery ─────────────────────────────────────────────────────────

def find_test_dirs(output_root):
    """Find all test output dirs sorted by test number."""
    pattern = os.path.join(output_root, "test_*")
    dirs = sorted(glob.glob(pattern))
    result = []
    for d in dirs:
        # Exclude structural, DTI-derived (_L1_AM, _L1_DT, _L1_DT_FA, _L1_DT_TR)
        # Keep only the main corrected 4D NIfTI
        niis = [f for f in glob.glob(os.path.join(d, "*.nii"))
                if "structural" not in os.path.basename(f)
                and "_L1_" not in os.path.basename(f)]
        structs = glob.glob(os.path.join(d, "structural_0.nii"))
        if niis and structs:
            name = os.path.basename(d)
            result.append((name, niis[0], structs[0]))
    return result


# ─── Per-volume DICE ────────────────────────────────────────────────────────

def analyze_test(name, nii_path, struct_path, focus_vol=40, verbose=False):
    """
    Compute per-volume DICE for one test.
    Returns: (name, mean_dice, min_dice, min_vol, focus_dice)
    """
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)          # (X, Y, Z, T)
    n_vols = data.shape[3] if data.ndim == 4 else 1

    struct_img = nib.load(struct_path)
    struct_data = struct_img.get_fdata(dtype=np.float32)
    ref_mask = brain_mask(struct_data)

    scores = np.zeros(n_vols)
    for v in range(n_vols):
        vol = data[..., v] if data.ndim == 4 else data
        vmask = brain_mask(vol)
        scores[v] = dice(vmask, ref_mask)

    mean_d = scores.mean()
    min_d = scores.min()
    min_v = int(np.argmin(scores))
    focus_d = scores[focus_vol] if focus_vol < n_vols else float("nan")

    if verbose:
        print(f"\n  {name}: per-volume DICE (all {n_vols} vols)")
        for v, s in enumerate(scores):
            marker = " <-- FOCUS" if v == focus_vol else ""
            marker += " <-- MIN" if v == min_v and v != focus_vol else ""
            if s < 0.80 or v == focus_vol or v == min_v:
                print(f"    vol {v:3d}: DICE={s:.4f}{marker}")

    return name, mean_d, min_d, min_v, focus_d, scores


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    focus_vol = 40
    if "--vol" in args:
        idx = args.index("--vol")
        focus_vol = int(args[idx + 1])
        args = args[:idx] + args[idx+2:]

    if len(args) < 1:
        print(__doc__)
        sys.exit(1)

    remote_root = args[0]
    local_root  = args[1] if len(args) > 1 else None

    # Collect all tests
    all_tests = []
    if os.path.isdir(remote_root):
        all_tests += find_test_dirs(remote_root)
    if local_root and os.path.isdir(local_root):
        all_tests += find_test_dirs(local_root)

    if not all_tests:
        print("No test directories with NIfTI + structural found.")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"Per-volume DICE analysis  (focus volume: {focus_vol})")
    print(f"  Reference: each test's own structural_0.nii  (Otsu-thresholded)")
    print(f"  DWI volumes: Otsu-thresholded per volume")
    print(f"{'='*72}")
    print(f"\n{'Test':<40} {'Mean DICE':>10} {'Min DICE':>10} {'Min vol':>8} {'Vol {:d} DICE'.format(focus_vol):>12}")
    print("-" * 82)

    results = []
    for name, nii_path, struct_path in all_tests:
        try:
            r = analyze_test(name, nii_path, struct_path, focus_vol=focus_vol)
            results.append(r)
            name_s, mean_d, min_d, min_v, focus_d, _ = r
            print(f"{name_s:<40} {mean_d:>10.4f} {min_d:>10.4f} {min_v:>8d} {focus_d:>12.4f}")
        except Exception as e:
            print(f"{name:<40}  ERROR: {e}")

    # Summary: rank by focus volume DICE
    print(f"\n{'─'*82}")
    print(f"Ranked by vol {focus_vol} DICE (best first):")
    ranked = sorted(results, key=lambda x: -x[4])
    for i, (name, mean_d, min_d, min_v, focus_d, _) in enumerate(ranked, 1):
        print(f"  {i:2d}. {name:<40} vol{focus_vol}={focus_d:.4f}  mean={mean_d:.4f}")

    # Worst-volume breakdown for test 24 and best overall
    print(f"\n{'─'*82}")
    print(f"Detailed per-volume DICE for notable tests (vols with DICE < 0.80):\n")
    notable = {r[0] for r in ranked[:3]}  # top 3 by focus vol
    notable |= {r[0] for r in sorted(results, key=lambda x: -x[1])[:2]}  # top 2 by mean
    # Always include test 24
    notable |= {r[0] for r in results if "test_24" in r[0]}

    for name, nii_path, struct_path in all_tests:
        if os.path.basename(name) in notable or name in notable:
            try:
                analyze_test(name, nii_path, struct_path,
                             focus_vol=focus_vol, verbose=True)
            except Exception as e:
                print(f"  {name}: ERROR {e}")


if __name__ == "__main__":
    main()
