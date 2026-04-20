"""Run all pipeline integration tests in order.

Tests are sequential — later steps depend on outputs from earlier steps.
For example, test04-06 use the model trained by test03.

Core tests stop on first failure (later tests depend on earlier ones).
Optional tests (--include-optional) run independently and don't stop on failure.
"""

import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

CORE_TESTS = [
    # Utils
    "test_utils00_config_device_metrics.py",
    "test_utils01_supervision_metrics.py",
    "test_utils02_exploration.py",
    "test_utils03_scaffold.py",
    "test_utils04_keypoint.py",
    "test_utils05_langgraph.py",
    "test_utils06_progress.py",
    "test_utils07_paddle_bridge.py",
    "test_utils08_release.py",
    "test_utils09_yolo_io.py",
    # p00: Data Preparation
    "test_p00_data_prep.py",
    # p01-p04: Annotation pipeline (independent, service-optional)
    "test_p01_auto_annotate.py",
    "test_p02_annotation_quality.py",
    "test_p03_generative_augment.py",
    "test_p04_label_studio.py",
    # p05: Data
    "test_p05_create_data.py",
    "test_p05_detection_dataset.py",
    "test_p05_classification_dataset.py",
    "test_p05_segmentation_dataset.py",
    "test_p05_keypoint_dataset.py",
    "test_p05_coco_dataset.py",
    "test_p05_augmentation_preview.py",
    # p06: Models + Training (sequential — checkpoint needed by later tests)
    "test_p06_model_registry.py",
    "test_p06_model_variants.py",
    "test_p06_training.py",
    "test_p06_training_hf_detection.py",
    "test_p06_training_features.py",
    "test_p06_classification_training.py",
    "test_p06_segmentation_metrics.py",
    "test_p06_segmentation_training.py",
    "test_p06_val_prediction_logger.py",
    # p07: HPO
    "test_p07_hpo.py",
    # p08: Evaluation (depends on p06 checkpoint)
    "test_p08_evaluation.py",
    "test_p08_error_analysis.py",
    # p09: Export (depends on p06 checkpoint)
    "test_p09_export.py",
    "test_p09_export_validation.py",
    # p10: Inference (depends on p06 + p09)
    "test_p10_inference.py",
    "test_p10_video_inference.py",
    "test_p10_face_recognition.py",
    # p11: End-to-end pipeline
    "test_p11_e2e_pipeline.py",
    # p12: Raw pipeline (raw images → auto-annotate → QA → train → eval → export → infer)
    "test_p12_raw_pipeline.py",
]

OPTIONAL_TESTS = []


def run_test(test_file):
    """Run a single test file and return (status, elapsed, returncode)."""
    test_path = TESTS_DIR / test_file
    if not test_path.exists():
        print(f"\n  SKIP: {test_file} (not found)")
        return "SKIP", 0, 0

    print(f"\n{'─' * 70}")
    print(f"  Running: {test_file}")
    print(f"{'─' * 70}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=str(TESTS_DIR.parent),
    )
    elapsed = time.time() - start

    status = "PASS" if result.returncode == 0 else "FAIL"
    return status, elapsed, result.returncode


def main():
    include_optional = "--include-optional" in sys.argv

    print("=" * 70)
    print("  CAMERA EDGE — Full Pipeline Integration Tests")
    if include_optional:
        print("  (including optional tests)")
    print("=" * 70)

    core_results = []
    optional_results = []
    total_start = time.time()

    # --- Core tests: stop on first failure ---
    core_stopped = False
    for test_file in CORE_TESTS:
        status, elapsed, returncode = run_test(test_file)
        core_results.append((test_file, status, elapsed))

        if status == "FAIL":
            print(f"\n  FAILED: {test_file} (exit code {returncode})")
            print(f"  Stopping — later tests depend on this step.")
            core_stopped = True
            break

    # --- Optional tests: run all, don't stop on failure ---
    if include_optional and not core_stopped:
        print(f"\n{'=' * 70}")
        print("  OPTIONAL TESTS")
        print(f"{'=' * 70}")

        for test_file in OPTIONAL_TESTS:
            status, elapsed, returncode = run_test(test_file)
            optional_results.append((test_file, status, elapsed))

            if status == "FAIL":
                print(f"\n  FAILED: {test_file} (exit code {returncode})")
                print(f"  Continuing with remaining optional tests...")

    total_elapsed = time.time() - total_start

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("  SUMMARY — Core Tests")
    print(f"{'=' * 70}")
    for test_file, status, elapsed in core_results:
        icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "○"
        print(f"  {icon} {test_file:<35s} {status:>5s}  ({elapsed:.1f}s)")

    core_not_run = len(CORE_TESTS) - len(core_results)
    if core_not_run > 0:
        for test_file in CORE_TESTS[len(core_results):]:
            print(f"  - {test_file:<35s}  NOT RUN")

    core_passed = sum(1 for _, s, _ in core_results if s == "PASS")
    core_failed = sum(1 for _, s, _ in core_results if s == "FAIL")
    core_skipped = sum(1 for _, s, _ in core_results if s == "SKIP")
    print(f"\n  Core: {core_passed} passed, {core_failed} failed, {core_skipped} skipped, {core_not_run} not run")

    if include_optional:
        print(f"\n{'=' * 70}")
        print("  SUMMARY — Optional Tests")
        print(f"{'=' * 70}")
        if optional_results:
            for test_file, status, elapsed in optional_results:
                icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "○"
                print(f"  {icon} {test_file:<35s} {status:>5s}  ({elapsed:.1f}s)")
        elif core_stopped:
            for test_file in OPTIONAL_TESTS:
                print(f"  - {test_file:<35s}  NOT RUN (core failed)")
        opt_passed = sum(1 for _, s, _ in optional_results if s == "PASS")
        opt_failed = sum(1 for _, s, _ in optional_results if s == "FAIL")
        opt_skipped = sum(1 for _, s, _ in optional_results if s == "SKIP")
        opt_not_run = len(OPTIONAL_TESTS) - len(optional_results)
        print(f"\n  Optional: {opt_passed} passed, {opt_failed} failed, {opt_skipped} skipped, {opt_not_run} not run")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Exit code based on core tests only — optional failures are informational
    if core_failed > 0:
        print("\n  CORE PIPELINE TESTS FAILED")
        sys.exit(1)
    else:
        if include_optional and any(s == "FAIL" for _, s, _ in optional_results):
            print("\n  CORE TESTS PASSED (some optional tests failed)")
        else:
            print("\n  ALL PIPELINE TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
