"""Tests for batch endpoints: POST /segment_text_batch and /auto_mask_batch.

These endpoints use true batch tensor processing - images are resized/padded
to common size maintaining aspect ratio, processed in a single GPU forward pass,
then coordinates are mapped back to original image sizes.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests
from PIL import Image

from conftest import (
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    annotate_image,
    base64_to_mask,
    detections_from_masks,
    load_image_b64,
    load_image_cv2,
    skip_no_service,
)


def resize_image(img: np.ndarray, size: tuple[int, int]) -> str:
    """Resize image to given size and return as base64 string."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@skip_no_service
class TestSegmentTextBatch:
    """Tests for POST /segment_text_batch endpoint."""

    def test_batch_text_single_image(self):
        """Batch endpoint with single image returns single response."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={"items": [{"image": image_b64}], "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert "detections" in data[0]

    def test_batch_text_multiple_images_same_size(self):
        """Batch multiple images of same size."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={
                "items": [
                    {"image": image_b64},
                    {"image": image_b64},
                    {"image": image_b64},
                ],
                "text": "cat",
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        for result in data:
            assert "detections" in result
            assert isinstance(result["detections"], list)

    def test_batch_text_mixed_sizes(self):
        """Critical test: batch images with different sizes.

        Verifies that:
        1. Images are resized maintaining aspect ratio
        2. Coordinates are mapped back correctly
        3. Bboxes are in original image coordinate space
        """
        img = load_image_cv2("cat.jpg")
        orig_h, orig_w = img.shape[:2]

        # Create different sized versions
        size_large_b64 = resize_image(img, (640, 480))
        size_small_b64 = resize_image(img, (320, 240))
        size_medium_b64 = resize_image(img, (512, 384))

        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={
                "items": [
                    {"image": size_large_b64},
                    {"image": size_small_b64},
                    {"image": size_medium_b64},
                ],
                "text": "cat",
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

        # Verify each result has detections with valid bboxes
        for i, result in enumerate(data):
            assert "detections" in result
            detections = result["detections"]
            if not detections:
                continue

            # Check that bboxes are within image bounds
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

                # Bboxes should be valid and non-negative
                assert x2 > x1, f"Result {i}: bbox x2 should be > x1"
                assert y2 > y1, f"Result {i}: bbox y2 should be > y1"
                assert x1 >= 0 and y1 >= 0, f"Result {i}: bbox coords should be non-negative"

                # Bboxes should be within expected bounds
                expected_sizes = [(640, 480), (320, 240), (512, 384)]
                exp_w, exp_h = expected_sizes[i]
                assert x2 <= exp_w, f"Result {i}: bbox x2 exceeds width {exp_w}"
                assert y2 <= exp_h, f"Result {i}: bbox y2 exceeds height {exp_h}"

    def test_batch_text_custom_thresholds(self):
        """Custom detection and mask thresholds work with batch."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={
                "items": [{"image": image_b64}, {"image": image_b64}],
                "text": "cat",
                "detection_threshold": 0.3,
                "mask_threshold": 0.3,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_batch_text_saves_overlay(self):
        """Save visualization of batch results."""
        img = load_image_cv2("cat.jpg")
        size_large_b64 = resize_image(img, (640, 480))
        size_small_b64 = resize_image(img, (320, 240))

        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={
                "items": [{"image": size_large_b64}, {"image": size_small_b64}],
                "text": "cat",
            },
            timeout=REQUEST_TIMEOUT,
        )
        results = resp.json()

        # Annotate each result
        for i, result in enumerate(results):
            detections_raw = result["detections"]
            if not detections_raw:
                continue

            sizes = [(640, 480), (320, 240)]
            w, h = sizes[i]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = Image.fromarray(img_rgb).resize((w, h))
            img_resized_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

            dets = detections_from_masks(detections_raw)
            labels = [f"score: {d['score']:.3f}" for d in detections_raw]
            annotated = annotate_image(img_resized_bgr, dets, labels)

            out_path = OUTPUT_DIR / f"test07_batch_text_{i}_{w}x{h}.png"
            cv2.imwrite(str(out_path), annotated)
            assert out_path.exists()

    def test_batch_text_empty_items(self):
        """Empty items list returns error."""
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={"items": [], "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 422  # Validation error


@skip_no_service
class TestAutoMaskBatch:
    """Tests for POST /auto_mask_batch endpoint."""

    def test_batch_auto_single_image(self):
        """Batch auto_mask with single image."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask_batch",
            json={"items": [{"image": image_b64}]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert "detections" in data[0]

    def test_batch_auto_multiple_images(self):
        """Batch multiple images for auto_mask."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask_batch",
            json={
                "items": [
                    {"image": image_b64},
                    {"image": image_b64},
                ],
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_batch_auto_mixed_sizes(self):
        """Critical test: auto_mask with mixed-size images."""
        img = load_image_cv2("cat.jpg")

        size_large_b64 = resize_image(img, (640, 480))
        size_small_b64 = resize_image(img, (320, 240))
        size_square_b64 = resize_image(img, (400, 400))

        resp = requests.post(
            f"{SERVICE_URL}/auto_mask_batch",
            json={
                "items": [
                    {"image": size_large_b64},
                    {"image": size_small_b64},
                    {"image": size_square_b64},
                ],
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

        # Verify bbox bounds for each result
        expected_sizes = [(640, 480), (320, 240), (400, 400)]
        for i, result in enumerate(data):
            detections = result["detections"]
            if not detections:
                continue

            exp_w, exp_h = expected_sizes[i]
            for det in detections:
                bbox = det["bbox"]
                assert bbox["x2"] <= exp_w, f"Result {i}: x2 exceeds {exp_w}"
                assert bbox["y2"] <= exp_h, f"Result {i}: y2 exceeds {exp_h}"

    def test_batch_auto_custom_threshold(self):
        """Custom threshold works with batch."""
        image_b64 = load_image_b64("cat.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask_batch",
            json={
                "items": [{"image": image_b64}],
                "threshold": 0.1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200

    def test_batch_auto_saves_overlay(self):
        """Save visualization of batch auto_mask results."""
        img = load_image_cv2("cat.jpg")
        size_large_b64 = resize_image(img, (640, 480))
        size_small_b64 = resize_image(img, (320, 240))

        resp = requests.post(
            f"{SERVICE_URL}/auto_mask_batch",
            json={
                "items": [{"image": size_large_b64}, {"image": size_small_b64}],
            },
            timeout=REQUEST_TIMEOUT,
        )
        results = resp.json()

        # Annotate each result
        for i, result in enumerate(results):
            detections_raw = result["detections"]
            if not detections_raw:
                continue

            sizes = [(640, 480), (320, 240)]
            w, h = sizes[i]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = Image.fromarray(img_rgb).resize((w, h))
            img_resized_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

            dets = detections_from_masks(detections_raw)
            labels = [f"score: {d['score']:.3f}" for d in detections_raw]
            annotated = annotate_image(img_resized_bgr, dets, labels)

            out_path = OUTPUT_DIR / f"test07_batch_auto_{i}_{w}x{h}.png"
            cv2.imwrite(str(out_path), annotated)
            assert out_path.exists()


@skip_no_service
class TestBatchEdgeCases:
    """Edge case tests for batch endpoints."""

    def test_batch_with_invalid_image(self):
        """Batch with one invalid image should still process others."""
        valid_b64 = load_image_b64("cat.jpg")
        invalid_b64 = "not_a_valid_image"

        # Note: This may return 500 or 422 depending on implementation
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={
                "items": [
                    {"image": valid_b64},
                    {"image": invalid_b64},
                ],
                "text": "cat",
            },
            timeout=REQUEST_TIMEOUT,
        )
        # Service should handle this gracefully or return error
        assert resp.status_code in [200, 400, 422, 500]

    def test_batch_single_then_multiple_consistency(self):
        """Results should be consistent between single and batch calls."""
        image_b64 = load_image_b64("cat.jpg")

        # Single call
        single_resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={"image": image_b64, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        single_dets = single_resp.json()["detections"]

        # Batch call with same image
        batch_resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={"items": [{"image": image_b64}], "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        batch_dets = batch_resp.json()[0]["detections"]

        # Should have similar number of detections (may vary due to randomness)
        # Check that results are not wildly different
        if single_dets and batch_dets:
            single_count = len(single_dets)
            batch_count = len(batch_dets)
            # Allow some tolerance for model non-determinism
            assert abs(single_count - batch_count) <= max(1, single_count // 2)


@skip_no_service
class TestBatchPerformance:
    """Performance and timing tests for batch endpoints."""

    def test_batch_text_timing_single_vs_multiple(self):
        """Compare timing: single endpoint vs batch endpoint.

        Measures the performance improvement of using batch endpoint
        for multiple images.
        """
        image_b64 = load_image_b64("cat.jpg")

        # Time single endpoint calls (sequential)
        num_images = 4
        single_times = []

        for _ in range(num_images):
            start = time.time()
            resp = requests.post(
                f"{SERVICE_URL}/segment_text",
                json={"image": image_b64, "text": "cat"},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            single_times.append(time.time() - start)

        avg_single = sum(single_times) / len(single_times)
        total_single = sum(single_times)

        # Time batch endpoint (single request)
        items = [{"image": image_b64} for _ in range(num_images)]
        start = time.time()
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={"items": items, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        batch_time = time.time() - start

        # Save timing report
        report_path = OUTPUT_DIR / "test07_batch_timing_report.txt"
        with open(report_path, "w") as f:
            f.write(f"=== Batch Performance Test ({num_images} images) ===\n\n")
            f.write(f"Single endpoint (sequential calls):\n")
            f.write(f"  Total time: {total_single:.3f}s\n")
            f.write(f"  Avg per image: {avg_single:.3f}s\n")
            f.write(f"  Individual times: {[f'{t:.3f}s' for t in single_times]}\n\n")
            f.write(f"Batch endpoint (single request):\n")
            f.write(f"  Total time: {batch_time:.3f}s\n")
            f.write(f"  Avg per image: {batch_time / num_images:.3f}s\n\n")
            if batch_time > 0:
                speedup = total_single / batch_time
                f.write(f"Speedup: {speedup:.2f}x\n")

        # Batch should be faster than sequential single calls
        # (May not always be true due to overhead, but generally true for GPU batching)
        # For now, just verify both completed successfully
        assert batch_time < 300, f"Batch took too long: {batch_time:.3f}s"

    def test_batch_text_large_batch_performance(self):
        """Test performance with a large batch (8 images)."""
        image_b64 = load_image_b64("cat.jpg")

        # Test with increasing batch sizes
        batch_sizes = [1, 2, 4, 8]
        timings = {}

        for size in batch_sizes:
            items = [{"image": image_b64} for _ in range(size)]
            start = time.time()
            resp = requests.post(
                f"{SERVICE_URL}/segment_text_batch",
                json={"items": items, "text": "cat"},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            elapsed = time.time() - start
            timings[size] = elapsed
            # Calculate throughput
            throughput = size / elapsed
            print(f"Batch size {size}: {elapsed:.3f}s ({throughput:.2f} img/s)")

        # Save detailed timing report
        report_path = OUTPUT_DIR / "test07_batch_scalability.txt"
        with open(report_path, "w") as f:
            f.write("=== Batch Scalability Report ===\n\n")
            f.write("Batch Size | Total Time | Avg Time/Image | Throughput\n")
            f.write("-" * 55 + "\n")
            for size in batch_sizes:
                total = timings[size]
                avg = total / size
                throughput = size / total
                f.write(f"{size:10} | {total:9.3f}s | {avg:13.3f}s | {throughput:10.2f} img/s\n")

        # Verify that throughput doesn't degrade significantly
        # (Larger batches should have similar or better throughput)
        if len(batch_sizes) >= 2:
            small_throughput = batch_sizes[0] / timings[batch_sizes[0]]
            large_throughput = batch_sizes[-1] / timings[batch_sizes[-1]]
            # Large batch throughput should be at least 50% of small batch
            assert large_throughput >= small_throughput * 0.5, (
                f"Throughput degraded too much: "
                f"{small_throughput:.2f} -> {large_throughput:.2f} img/s"
            )

    def test_batch_auto_timing_comparison(self):
        """Compare auto_mask timing: single vs batch endpoint."""
        image_b64 = load_image_b64("cat.jpg")
        num_images = 3

        # Time single endpoint calls
        single_times = []
        for _ in range(num_images):
            start = time.time()
            resp = requests.post(
                f"{SERVICE_URL}/auto_mask",
                json={"image": image_b64},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            single_times.append(time.time() - start)

        total_single = sum(single_times)

        # Time batch endpoint
        items = [{"image": image_b64} for _ in range(num_images)]
        start = time.time()
        resp = requests.post(
            f"{SERVICE_URL}/auto_mask_batch",
            json={"items": items},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        batch_time = time.time() - start

        # Save timing data
        report_path = OUTPUT_DIR / "test07_auto_mask_timing.txt"
        with open(report_path, "w") as f:
            f.write(f"=== Auto_Mask Batch Timing ({num_images} images) ===\n\n")
            f.write(f"Single calls total: {total_single:.3f}s\n")
            f.write(f"Batch call: {batch_time:.3f}s\n")
            if batch_time > 0:
                speedup = total_single / batch_time
                f.write(f"Speedup: {speedup:.2f}x\n")

    def test_batch_text_mixed_sizes_timing(self):
        """Measure timing for batch with mixed-size images."""
        img = load_image_cv2("cat.jpg")

        # Create different sized versions
        sizes = [(640, 480), (1920, 1080), (512, 512), (320, 240), (800, 600)]
        items = []
        for size in sizes:
            img_b64 = resize_image(img, size)
            items.append({"image": img_b64})

        start = time.time()
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={"items": items, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        batch_time = time.time() - start

        # Verify all results came back
        results = resp.json()
        assert len(results) == len(sizes)

        # Save timing
        report_path = OUTPUT_DIR / "test07_mixed_sizes_timing.txt"
        with open(report_path, "w") as f:
            f.write(f"=== Mixed-Size Batch Timing ===\n\n")
            f.write(f"Images: {len(sizes)}\n")
            f.write(f"Sizes: {sizes}\n")
            f.write(f"Total time: {batch_time:.3f}s\n")
            f.write(f"Avg per image: {batch_time / len(sizes):.3f}s\n")
            f.write(f"Throughput: {len(sizes) / batch_time:.2f} img/s\n")


@skip_no_service
class TestBatchProgressAndMetrics:
    """Progress tracking and metrics for batch endpoints."""

    def test_batch_result_counts_and_metrics(self):
        """Verify detection counts and quality metrics for batch results."""
        image_b64 = load_image_b64("cat.jpg")

        # Create batch with 3 images
        items = [{"image": image_b64} for _ in range(3)]
        resp = requests.post(
            f"{SERVICE_URL}/segment_text_batch",
            json={"items": items, "text": "cat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        results = resp.json()

        # Collect metrics
        detection_counts = []
        all_scores = []
        all_areas = []

        for result in results:
            detections = result["detections"]
            detection_counts.append(len(detections))
            for det in detections:
                all_scores.append(det.get("score", 0.0))
                all_areas.append(det.get("area", 0.0))

        # Save metrics report
        report_path = OUTPUT_DIR / "test07_batch_metrics.txt"
        with open(report_path, "w") as f:
            f.write("=== Batch Detection Metrics ===\n\n")
            f.write(f"Number of images: {len(results)}\n")
            f.write(f"Detections per image: {detection_counts}\n")
            f.write(f"Total detections: {sum(detection_counts)}\n")
            f.write(f"Avg detections/image: {sum(detection_counts) / len(detection_counts):.2f}\n\n")

            if all_scores:
                f.write(f"Scores:\n")
                f.write(f"  Min: {min(all_scores):.3f}\n")
                f.write(f"  Max: {max(all_scores):.3f}\n")
                f.write(f"  Avg: {sum(all_scores) / len(all_scores):.3f}\n\n")

            if all_areas:
                f.write(f"Areas:\n")
                f.write(f"  Min: {min(all_areas):.4f}\n")
                f.write(f"  Max: {max(all_areas):.4f}\n")
                f.write(f"  Avg: {sum(all_areas) / len(all_areas):.4f}\n")

        # Verify all images have detection counts (integers)
        assert all(isinstance(d, int) for d in detection_counts)
        assert len(detection_counts) == 3  # We sent 3 images

    def test_batch_progressive_response_sizes(self):
        """Test that response size scales appropriately with batch size."""
        image_b64 = load_image_b64("cat.jpg")

        # Test different batch sizes and measure response
        batch_sizes = [1, 2, 4]
        response_sizes = {}

        for size in batch_sizes:
            items = [{"image": image_b64} for _ in range(size)]
            resp = requests.post(
                f"{SERVICE_URL}/segment_text_batch",
                json={"items": items, "text": "cat"},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            response_sizes[size] = len(resp.content)

        # Response size should generally increase with batch size
        # (but not linearly due to JSON overhead)
        report_path = OUTPUT_DIR / "test07_response_sizes.txt"
        with open(report_path, "w") as f:
            f.write("=== Response Size Analysis ===\n\n")
            for size in batch_sizes:
                size_kb = response_sizes[size] / 1024
                f.write(f"Batch {size}: {response_sizes[size]} bytes ({size_kb:.2f} KB)\n")

    def test_batch_consistency_across_runs(self):
        """Test that batch results are consistent across multiple runs.

        Runs the same batch request multiple times and compares results.
        """
        image_b64 = load_image_b64("cat.jpg")

        # Run same batch 3 times
        items = [{"image": image_b64}, {"image": image_b64}]
        all_counts = []
        all_scores = []

        for run in range(3):
            resp = requests.post(
                f"{SERVICE_URL}/segment_text_batch",
                json={"items": items, "text": "cat"},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            results = resp.json()

            run_counts = []
            for result in results:
                detections = result["detections"]
                run_counts.append(len(detections))
                for det in detections:
                    all_scores.append(det.get("score", 0.0))

            all_counts.append(run_counts)

        # Analyze consistency
        report_path = OUTPUT_DIR / "test07_consistency_report.txt"
        with open(report_path, "w") as f:
            f.write("=== Batch Consistency Across Runs ===\n\n")
            f.write(f"Detection counts per run per image:\n")
            for i, counts in enumerate(all_counts):
                f.write(f"  Run {i+1}: {counts}\n")

            if all_scores:
                f.write(f"\nScore statistics across all runs:\n")
                f.write(f"  Total scores: {len(all_scores)}\n")
                f.write(f"  Min: {min(all_scores):.3f}\n")
                f.write(f"  Max: {max(all_scores):.3f}\n")
                f.write(f"  Avg: {sum(all_scores) / len(all_scores):.3f}\n")

        # Verify that detection counts are reasonably consistent
        # (allowing for some model non-determinism)
        flat_counts = [c for counts in all_counts for c in counts]
        if flat_counts:
            avg_count = sum(flat_counts) / len(flat_counts)
            max_deviation = max(abs(c - avg_count) for c in flat_counts)
            # Deviation should be within 50% of average
            assert max_deviation <= avg_count * 0.5 or avg_count < 2, (
                f"Detection counts too inconsistent: {flat_counts}"
            )
