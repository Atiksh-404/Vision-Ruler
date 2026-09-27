"""Synthetic regression coverage for target contour scoring."""

import cv2
import numpy as np

from main import (TargetTracker, score_candidate_contour, measure_target_object,
                  _likely_skin_mask, segment_center_object)


def _rect_contour(width, height, angle=0, center=(300, 220)):
    rect = (center, (float(width), float(height)), float(angle))
    return cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.int32)


def test_pen_like_contours_pass_without_opening_noise_filter():
    edges = np.zeros((440, 600), dtype=np.uint8)
    pen = _rect_contour(270, 9, 23)
    cv2.drawContours(edges, [pen], -1, 255, 2)

    score, touching = score_candidate_contour(pen, 600, 440, edges=edges)
    assert not touching
    assert score > 0.15
    tracker = TargetTracker(persistence_frames=3)
    for _ in range(3):
        target, _, _ = tracker.update([(pen, score, False)], 600, 440)
    assert target is not None
    assert tracker.current_dimensions is not None

    # A long, zero-thickness hair/glare line does not have a filled silhouette.
    hair = np.array([[[100, 100]], [[500, 101]], [[500, 102]], [[100, 101]]], dtype=np.int32)
    assert score_candidate_contour(hair, 600, 440, edges=edges)[0] == 0.0


def test_existing_broad_object_shapes_remain_eligible():
    for width, height, angle in ((180, 100, 0), (210, 140, 8), (90, 230, -12)):
        contour = _rect_contour(width, height, angle)
        score, touching = score_candidate_contour(contour, 600, 440)
        assert not touching
        assert score > 0.15


def test_min_area_rect_dimensions_keep_pen_long_and_short_sides():
    pen = _rect_contour(270, 9, 23)
    result = measure_target_object(pen, None, 0, 0)
    assert result is not None
    width_px, height_px = result["w_px"], result["h_px"]
    assert width_px > height_px
    assert width_px >= 260
    assert 6 <= height_px <= 12


def test_skin_suppression_targets_hand_tones_without_masking_blue_object():
    scene = np.zeros((80, 120, 3), dtype=np.uint8)
    scene[:] = (30, 30, 30)
    scene[20:70, 40:75] = (180, 45, 20)  # blue bottle body
    scene[35:55, 30:42] = (90, 140, 220)  # warm hand/finger region
    skin = _likely_skin_mask(scene)
    assert np.count_nonzero(skin[35:55, 30:42]) > 0
    assert np.count_nonzero(skin[20:70, 45:70]) == 0


def test_segmentation_resizes_skin_mask_to_original_crop():
    scene = np.full((240, 320, 3), 220, dtype=np.uint8)
    cv2.rectangle(scene, (130, 45), (190, 205), (180, 45, 20), -1)
    mask, _ = segment_center_object(scene, scale=0.60)
    assert mask.shape == scene.shape[:2]


