import cv2
import numpy as np
import ctypes
import os
import time
from collections import deque

# ---------- Configuration ----------

ROI_WIDTH_RATIO = 0.65   # 65% of camera width
ROI_HEIGHT_RATIO = 0.65  # 65% of camera height

MIN_CONTOUR_AREA = 500   # Base minimum area in pixels
CLOSING_KERNEL_SIZE = 5
MIN_TARGET_ROI_AREA_RATIO = 0.035
MIN_TARGET_BBOX_ROI_RATIO = 0.035
TARGET_CONFIRM_FRAMES = 8
TARGET_LOST_FRAMES = 10
TARGET_SEARCH_MARGIN_RATIO = 0.65
DEBUG_TARGET_DETECTION = True
SEGMENTATION_INTERVAL = 4
SEGMENTATION_SCALE = 0.60
SEGMENTATION_PADDING = 25
TRACKING_MATCH_SCALE = 0.25
TRACKING_EARLY_MOVE_PIXELS = 18
PERF_REPORT_INTERVAL = 60

# ISO/IEC 7810 ID-1 standard bank/credit card dimensions:
REFERENCE_WIDTH_CM = 8.56    # cm
REFERENCE_HEIGHT_CM = 5.398  # cm
REFERENCE_ASPECT_RATIO = REFERENCE_WIDTH_CM / REFERENCE_HEIGHT_CM  # ~1.5857
REFERENCE_MIN_ROI_AREA_RATIO = 0.035

CALIBRATION_SAMPLES_NEEDED = 8
CALIBRATION_TIMEOUT_FRAMES = 150  # ~5 seconds at 30 fps


class TargetTracker:
    """
    Confirms one spatially consistent object, locks it, and holds through brief misses.
    """
    def __init__(self, persistence_frames=TARGET_CONFIRM_FRAMES, lost_frames=TARGET_LOST_FRAMES):
        self.current_bbox = None
        self.current_contour = None
        self.current_center = None
        self.current_area = 0.0
        self.current_dimensions = None
        self.current_mask = None
        self.previous_bbox = None
        self.current_score = 0.0
        self.missed_frames = 0
        self.persistence = persistence_frames
        self.lost_limit = lost_frames
        self.pending_bbox = None
        self.pending_contour = None
        self.pending_frames = 0
        self.state = "NO_TARGET"

    @staticmethod
    def _bbox_metrics(first_bbox, second_bbox):
        ax, ay, aw, ah = first_bbox
        bx, by, bw, bh = second_bbox
        intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
            0, min(ay + ah, by + bh) - max(ay, by)
        )
        iou = intersection / max(aw * ah + bw * bh - intersection, 1)
        center_a = (ax + aw / 2.0, ay + ah / 2.0)
        center_b = (bx + bw / 2.0, by + bh / 2.0)
        diagonal = max(float(np.hypot(max(aw, bw), max(ah, bh))), 1.0)
        center_similarity = max(0.0, 1.0 - np.hypot(
            center_a[0] - center_b[0], center_a[1] - center_b[1]
        ) / diagonal)
        area_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh, 1)
        direct_dims = min(aw / max(bw, 1), bw / max(aw, 1)) * min(
            ah / max(bh, 1), bh / max(ah, 1)
        )
        swapped_dims = min(aw / max(bh, 1), bh / max(aw, 1)) * min(
            ah / max(bw, 1), bw / max(ah, 1)
        )
        return iou, center_similarity, area_ratio, max(direct_dims, swapped_dims)

    @staticmethod
    def _shape_similarity(first_contour, second_contour):
        try:
            return cv2.matchShapes(first_contour, second_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        except cv2.error:
            return 1.0

    def _candidate_consistent(self, candidate, reference_bbox, reference_contour, confirming=False):
        contour, _, _ = candidate
        bbox = cv2.boundingRect(contour)
        iou, center_similarity, area_ratio, dimension_similarity = self._bbox_metrics(
            bbox, reference_bbox
        )
        shape_distance = self._shape_similarity(contour, reference_contour)
        if confirming:
            return (iou >= 0.25 or (
                center_similarity >= 0.78 and area_ratio >= 0.55 and dimension_similarity >= 0.45
            )) and shape_distance <= 0.65
        return (iou >= 0.08 or (
            center_similarity >= 0.48 and area_ratio >= 0.40 and dimension_similarity >= 0.35
        )) and shape_distance <= 1.0

    def _begin_confirmation(self, candidate):
        contour, score, _ = candidate
        self.pending_bbox = cv2.boundingRect(contour)
        self.pending_contour = contour
        self.pending_frames = 1
        self.current_score = score
        self.state = "CANDIDATE_DETECTED"

    def _set_current_target(self, contour, score, roi_w, roi_h):
        self.previous_bbox = self.current_bbox
        self.current_bbox = cv2.boundingRect(contour)
        x, y, width, height = self.current_bbox
        self.current_center = (x + width / 2.0, y + height / 2.0)
        self.current_area = float(cv2.contourArea(contour))
        rect = cv2.minAreaRect(contour)
        self.current_dimensions = tuple(float(value) for value in rect[1])
        self.current_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.drawContours(self.current_mask, [contour], -1, 255, thickness=cv2.FILLED)
        self.current_contour = contour
        self.current_score = score

    def _clear_current_target(self):
        self.current_bbox = None
        self.current_contour = None
        self.current_center = None
        self.current_area = 0.0
        self.current_dimensions = None
        self.current_mask = None
        self.previous_bbox = None
        self.current_score = 0.0

    def _format_state(self):
        if self.state == "CONFIRMING":
            return f"CONFIRMING {self.pending_frames}/{self.persistence}"
        if self.state == "TARGET_LOCKED":
            return "TARGET_LOCKED"
        if self.state == "TARGET_TRACKING":
            return "TARGET_TRACKING"
        if self.state == "TARGET_LOST":
            return "TARGET_TRACKING_HOLDING"
        return "SEARCHING"

    def update(self, scored_candidates, roi_w, roi_h):
        valid_candidates = [candidate for candidate in scored_candidates
                            if not candidate[2] and candidate[1] > 0.15]
        edge_candidates = [candidate for candidate in scored_candidates if candidate[2]]

        if self.current_contour is not None:
            matching = [candidate for candidate in valid_candidates if self._candidate_consistent(
                candidate, self.current_bbox, self.current_contour
            )]
            if matching:
                matching.sort(key=lambda candidate: (
                    self._bbox_metrics(cv2.boundingRect(candidate[0]), self.current_bbox)[0] * 0.50
                    + self._bbox_metrics(cv2.boundingRect(candidate[0]), self.current_bbox)[1] * 0.25
                    + self._bbox_metrics(cv2.boundingRect(candidate[0]), self.current_bbox)[2] * 0.15
                    + candidate[1] * 0.10
                ), reverse=True)
                contour, score, _ = matching[0]
                self._set_current_target(contour, score, roi_w, roi_h)
                self.missed_frames = 0
                self.state = "TARGET_TRACKING"
                return contour, False, self._format_state()

            matching_edges = [candidate for candidate in edge_candidates if self._candidate_consistent(
                candidate, self.current_bbox, self.current_contour
            )]
            if not matching_edges:
                matching_edges = [candidate for candidate in edge_candidates if (
                    self._bbox_metrics(cv2.boundingRect(candidate[0]), self.current_bbox)[0] >= 0.03
                    or self._bbox_metrics(cv2.boundingRect(candidate[0]), self.current_bbox)[1] >= 0.55
                )]
            self.missed_frames += 1
            if matching_edges:
                self.state = "TARGET_LOST"
                return None, True, "EDGE_WARNING"
            if self.missed_frames < self.lost_limit:
                self.state = "TARGET_LOST"
                return self.current_contour, False, self._format_state()

            self._clear_current_target()
            self.missed_frames = 0
            self.pending_bbox = None
            self.pending_contour = None
            self.pending_frames = 0
            self.state = "NO_TARGET"

        if not valid_candidates:
            self.pending_bbox = None
            self.pending_contour = None
            self.pending_frames = 0
            self.state = "NO_TARGET"
            if edge_candidates:
                return None, True, "EDGE_WARNING"
            return None, False, "UNCLEAR" if scored_candidates else "SEARCHING"

        if self.pending_bbox is None:
            valid_candidates.sort(key=lambda candidate: candidate[1], reverse=True)
            self._begin_confirmation(valid_candidates[0])
        else:
            consistent = [candidate for candidate in valid_candidates if self._candidate_consistent(
                candidate, self.pending_bbox, self.pending_contour, confirming=True
            )]
            if not consistent:
                valid_candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                self.pending_bbox = None
                self.pending_contour = None
                self.pending_frames = 0
                self._begin_confirmation(valid_candidates[0])
            else:
                consistent.sort(key=lambda candidate: candidate[1], reverse=True)
                contour, score, _ = consistent[0]
                self.pending_bbox = cv2.boundingRect(contour)
                self.pending_contour = contour
                self.pending_frames += 1
                self.current_score = score

        if self.pending_frames >= self.persistence:
            self._set_current_target(self.pending_contour, self.current_score, roi_w, roi_h)
            self.pending_bbox = None
            self.pending_contour = None
            self.pending_frames = 0
            self.missed_frames = 0
            self.state = "TARGET_LOCKED"
            return self.current_contour, False, self._format_state()

        self.state = "CONFIRMING"
        return None, False, self._format_state()

    def reset(self):
        self._clear_current_target()
        self.missed_frames = 0
        self.pending_bbox = None
        self.pending_contour = None
        self.pending_frames = 0
        self.state = "NO_TARGET"


class MeasurementStabilizer:
    """
    Applies temporal median filtering to width and height measurements.
    """
    def __init__(self, history_len=7):
        self.history = deque(maxlen=history_len)
        self.last_valid = None

    def update(self, w_cm, h_cm):
        if w_cm <= 0 or h_cm <= 0 or np.isnan(w_cm) or np.isnan(h_cm):
            return self.last_valid

        # Reset history on sudden major change (> 40%) so transitions are instantaneous
        if self.history:
            prev_w, prev_h = self.history[-1]
            if abs(w_cm - prev_w) > max(prev_w * 0.40, 2.5) or abs(h_cm - prev_h) > max(prev_h * 0.40, 2.5):
                self.history.clear()

        self.history.append((w_cm, h_cm))

        widths = [m[0] for m in self.history]
        heights = [m[1] for m in self.history]

        med_w = float(np.median(widths))
        med_h = float(np.median(heights))

        self.last_valid = (med_w, med_h)
        return self.last_valid

    def get_variance(self):
        if len(self.history) < 3:
            return 999.0
        widths = [m[0] for m in self.history]
        heights = [m[1] for m in self.history]
        return float(np.var(widths) + np.var(heights))

    def reset(self):
        self.history.clear()
        self.last_valid = None


def get_screen_resolution():
    try:
        user32 = ctypes.windll.user32
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        if sw > 0 and sh > 0:
            return sw, sh
    except Exception:
        pass
    return 1280, 720


def get_focus_roi(frame, width_ratio=ROI_WIDTH_RATIO, height_ratio=ROI_HEIGHT_RATIO):
    frame_h, frame_w = frame.shape[:2]
    roi_w = int(frame_w * width_ratio)
    roi_h = int(frame_h * height_ratio)
    roi_x = (frame_w - roi_w) // 2
    roi_y = (frame_h - roi_h) // 2

    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    return roi, (roi_x, roi_y, roi_w, roi_h)


def process_roi_outer_silhouettes(roi, search_bbox=None, return_debug=False):
    
    roi_h, roi_w = roi.shape[:2]
    crop_origin = (0, 0)
    working_roi = roi
    if search_bbox is not None:
        bx, by, bw, bh = search_bbox
        margin = max(24, int(max(bw, bh) * TARGET_SEARCH_MARGIN_RATIO))
        x0 = max(0, bx - margin)
        y0 = max(0, by - margin)
        x1 = min(roi_w, bx + bw + margin)
        y1 = min(roi_h, by + bh + margin)
        if x1 > x0 and y1 > y0:
            crop_origin = (x0, y0)
            working_roi = roi[y0:y1, x0:x1]

    gray = cv2.cvtColor(working_roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    edges = cv2.Canny(blurred, 40, 120)

    kernel_size = max(3, int(CLOSING_KERNEL_SIZE) | 1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    clean_edges = closed_edges

    contours, _ = cv2.findContours(clean_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_level_contours = list(contours)

    _, otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    block_size = min(31, min(norm.shape[:2]) | 1)
    block_size = max(3, block_size)
    adaptive = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 3
    )
    threshold_images = (otsu, cv2.bitwise_not(otsu), adaptive, cv2.bitwise_not(adaptive))
    min_threshold_area = max(160, int(roi_w * roi_h * 0.001))
    for binary in threshold_images:
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        threshold_contours, _ = cv2.findContours(
            closed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in threshold_contours:
            tx, ty, tw, th = cv2.boundingRect(contour)
            touches_working_edge = (
                tx <= 2 or ty <= 2 or tx + tw >= working_roi.shape[1] - 2
                or ty + th >= working_roi.shape[0] - 2
            )
            if cv2.contourArea(contour) >= min_threshold_area and not touches_working_edge:
                top_level_contours.append(contour)

    crop_x, crop_y = crop_origin
    crop_h, crop_w = clean_edges.shape[:2]
    full_edges = np.zeros((roi_h, roi_w), dtype=np.uint8)
    full_edges[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = clean_edges

    translated_contours = []
    for contour in top_level_contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        crop_touches_border = (
            cx <= 2 or cy <= 2 or cx + cw >= crop_w - 2 or cy + ch >= crop_h - 2
        )
        actual_roi_border = (
            (crop_x == 0 and cx <= 2) or
            (crop_y == 0 and cy <= 2) or
            (crop_x + crop_w == roi_w and cx + cw >= crop_w - 2) or
            (crop_y + crop_h == roi_h and cy + ch >= crop_h - 2)
        )
        if search_bbox is not None and crop_touches_border and not actual_roi_border:
            continue
        contour = contour.copy()
        contour[:, 0, 0] += crop_x
        contour[:, 0, 1] += crop_y
        translated_contours.append(contour)

    if return_debug:
        full_gray = np.zeros((roi_h, roi_w), dtype=np.uint8)
        full_blur = np.zeros((roi_h, roi_w), dtype=np.uint8)
        full_canny = np.zeros((roi_h, roi_w), dtype=np.uint8)
        full_gray[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = gray
        full_blur[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = blurred
        full_canny[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = edges
        debug_images = {
            "gray": full_gray,
            "blur": full_blur,
            "canny": full_canny,
            "morphology": full_edges,
        }
        return full_edges, translated_contours, debug_images

    return full_edges, translated_contours


def segment_center_object(roi, search_bbox=None, scale=SEGMENTATION_SCALE,
                          padding=SEGMENTATION_PADDING):
    """Segment at reduced resolution, optionally around the locked target, then restore ROI coordinates."""
    roi_h, roi_w = roi.shape[:2]
    crop_x = crop_y = 0
    working_roi = roi
    if search_bbox is not None:
        bx, by, bw, bh = search_bbox
        x0 = max(0, bx - padding)
        y0 = max(0, by - padding)
        x1 = min(roi_w, bx + bw + padding)
        y1 = min(roi_h, by + bh + padding)
        if x1 > x0 and y1 > y0:
            crop_x, crop_y = x0, y0
            working_roi = roi[y0:y1, x0:x1]

    local_h, local_w = working_roi.shape[:2]
    scale = min(1.0, max(0.15, float(scale)))
    work_w = max(32, int(round(local_w * scale)))
    work_h = max(32, int(round(local_h * scale)))
    working = cv2.resize(working_roi, (work_w, work_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else working_roi
    skin_mask = _likely_skin_mask(working)

    margin_x = max(2, int(work_w * 0.03))
    margin_y = max(2, int(work_h * 0.03))
    rect = (margin_x, margin_y, work_w - 2 * margin_x, work_h - 2 * margin_y)
    if rect[2] < 16 or rect[3] < 16:
        return np.zeros((roi_h, roi_w), dtype=np.uint8), None

    gc_mask = np.zeros((work_h, work_w), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(working, gc_mask, rect, background_model, foreground_model,
                1, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((roi_h, roi_w), dtype=np.uint8), None

    foreground = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    # GrabCut commonly joins a held object to the fingers around it. Remove
    # likely exposed skin before component selection so the hand cannot become
    # part of the measured silhouette. Reapply after morphology, which can
    # otherwise close the gap back across a finger/object boundary.
    foreground[skin_mask > 0] = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    foreground[skin_mask > 0] = 0

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    min_component_area = max(64, int(work_w * work_h * 0.012))
    rect_x, rect_y, rect_w, rect_h = rect
    center_x, center_y = work_w / 2.0, work_h / 2.0
    center_scale = max(float(np.hypot(work_w, work_h)) * 0.30, 1.0)
    best_label = None
    best_component_score = -1.0

    for label in range(1, component_count):
        x, y, width, height, area = stats[label]
        if area < min_component_area:
            continue
        if x <= rect_x + 1 or y <= rect_y + 1 or x + width >= rect_x + rect_w - 1 or y + height >= rect_y + rect_h - 1:
            continue
        center_distance = np.hypot(centroids[label][0] - center_x,
                                   centroids[label][1] - center_y)
        if center_distance > center_scale:
            continue
        component_score = area * (1.0 - 0.45 * center_distance / center_scale)
        if component_score > best_component_score:
            best_component_score = component_score
            best_label = label

    if best_label is None:
        return np.zeros((roi_h, roi_w), dtype=np.uint8), None

    component = np.where(labels == best_label, 255, 0).astype(np.uint8)
    if scale < 1.0:
        component = cv2.resize(component, (local_w, local_h), interpolation=cv2.INTER_NEAREST)
        skin_mask = cv2.resize(skin_mask, (local_w, local_h), interpolation=cv2.INTER_NEAREST)

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, cleanup_kernel, iterations=2)
    component[skin_mask > 0] = 0

    flood = component.copy()
    flood_mask = np.zeros((local_h + 2, local_w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    component = cv2.bitwise_or(component, holes)
    # Hole filling must not restore skin that was deliberately removed.
    component[skin_mask > 0] = 0

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        full_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        full_mask[crop_y:crop_y + local_h, crop_x:crop_x + local_w] = component
        return full_mask, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < local_w * local_h * 0.012:
        full_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        full_mask[crop_y:crop_y + local_h, crop_x:crop_x + local_w] = component
        return full_mask, None

    contour = contour.copy()
    contour[:, 0, 0] += crop_x
    contour[:, 0, 1] += crop_y
    full_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    full_mask[crop_y:crop_y + local_h, crop_x:crop_x + local_w] = component
    return full_mask, contour


def _likely_skin_mask(image):
    """Conservative YCrCb/HSV skin estimate used only to split hand-held objects."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, cr, cb = cv2.split(ycrcb)
    hue, saturation, value = cv2.split(hsv)
    chroma_skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)
    hue_skin = ((hue <= 25) | (hue >= 170)) & (saturation >= 35) & (value >= 40)
    mask = (chroma_skin & hue_skin).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)


def estimate_target_motion(previous_gray, current_gray, bbox, padding=SEGMENTATION_PADDING):
    """Match a reduced local template and return confidence plus ROI-pixel translation."""
    if previous_gray is None or bbox is None:
        return 0.0, 0.0, 0.0

    roi_h, roi_w = current_gray.shape[:2]
    bx, by, bw, bh = bbox
    scale = TRACKING_MATCH_SCALE
    small_w = max(32, int(round(roi_w * scale)))
    small_h = max(32, int(round(roi_h * scale)))
    previous_small = cv2.resize(previous_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    x, y = int(round(bx * scale)), int(round(by * scale))
    template_w = max(8, int(round(bw * scale)))
    template_h = max(8, int(round(bh * scale)))
    x2 = min(small_w, x + template_w)
    y2 = min(small_h, y + template_h)
    template = previous_small[max(0, y):y2, max(0, x):x2]
    if template.shape[0] < 8 or template.shape[1] < 8:
        return 0.0, 0.0, 0.0
    if float(np.std(template)) < 2.0:
        return 0.0, 0.0, 0.0

    margin = max(12, int(round(max(2 * padding, 48) * scale)))
    search_x = max(0, x - margin)
    search_y = max(0, y - margin)
    search_x2 = min(small_w, x2 + margin)
    search_y2 = min(small_h, y2 + margin)
    search = current_small[search_y:search_y2, search_x:search_x2]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0.0, 0.0, 0.0

    match = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(match)
    dx = (search_x + location[0] - x) / scale
    dy = (search_y + location[1] - y) / scale
    return float(confidence), float(dx), float(dy)


def quick_canny_preview(roi):
    """Refresh the Canny panel cheaply on locked frames without extracting contours."""
    roi_h, roi_w = roi.shape[:2]
    preview_scale = min(1.0, 420.0 / max(roi_h, roi_w))
    if preview_scale < 1.0:
        preview = cv2.resize(roi, None, fx=preview_scale, fy=preview_scale,
                             interpolation=cv2.INTER_AREA)
    else:
        preview = roi
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    if preview_scale < 1.0:
        edges = cv2.resize(edges, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return edges


def order_corners(pts):
    """Order 4 points: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_reference_corners(contour):
    """Extract ordered 4 corners from reference card contour."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * peri, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        raw_corners = approx.reshape(4, 2)
        has_approx_quad = True
    else:
        min_rect = cv2.minAreaRect(contour)
        raw_corners = cv2.boxPoints(min_rect)
        has_approx_quad = False

    ordered = order_corners(raw_corners)
    return ordered, has_approx_quad


def is_valid_reference(contour, roi_w, roi_h):
    
    area = cv2.contourArea(contour)
    if area < max(500, int(roi_w * roi_h * 0.0015)):
        return False, None, False, 0.0

    bx, by, bw, bh = cv2.boundingRect(contour)
    if bw < max(24, int(roi_w * 0.04)) or bh < max(18, int(roi_h * 0.04)):
        return False, None, False, 0.0

    # Reject contours touching Focus Zone border
    if bx <= 4 or by <= 4 or (bx + bw) >= (roi_w - 4) or (by + bh) >= (roi_h - 4):
        return False, None, False, 0.0

    rect_area = bw * bh
    rectangularity = area / max(rect_area, 1)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    hull_rectangularity = hull_area / max(rect_area, 1)
    min_reference_area = max(500, int(roi_w * roi_h * REFERENCE_MIN_ROI_AREA_RATIO))
    if (hull_area < min_reference_area or hull_rectangularity < 0.62 or solidity < 0.20):
        return False, None, False, 0.0

    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect < 1.05 or aspect > 2.55:
        return False, None, False, 0.0

    # Incomplete Canny/threshold outlines may have low fill but a card-shaped hull.
    corner_source = contour if rectangularity >= 0.42 else hull
    corners, has_approx_quad = get_reference_corners(corner_source)

    aspect_diff = abs(aspect - REFERENCE_ASPECT_RATIO)
    aspect_score = max(0.0, 1.0 - (aspect_diff / 0.95))
    rect_score = min(1.0, hull_rectangularity)
    corner_score = 1.0 if has_approx_quad else 0.75
    area_score = min(1.0, hull_area / max(roi_w * roi_h * 0.20, 1))
    border_distance = min(bx, by, roi_w - bx - bw, roi_h - by - bh)
    edge_score = min(1.0, border_distance / max(min(roi_w, roi_h) * 0.08, 1))

    total_score = (
        (area_score * 0.30)
        + (corner_score * 0.25)
        + (aspect_score * 0.20)
        + (rect_score * 0.20)
        + (edge_score * 0.05)
    )
    return True, corners, has_approx_quad, total_score


def find_reference_card(contours, roi_w, roi_h, with_status=False):
    """Scan Focus Zone contours to locate the best reference card candidate."""
    best_candidate = None
    best_score = -1.0
    rejected_candidate = False
    too_small_candidate = False

    for c in contours:
        valid, corners, has_quad, score = is_valid_reference(c, roi_w, roi_h)
        if valid and score > best_score:
            best_score = score
            best_candidate = (c, corners, has_quad)
        elif not valid:
            area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            away_from_border = (bx > 4 and by > 4 and bx + bw < roi_w - 4 and by + bh < roi_h - 4)
            plausible_card = (
                area >= max(250, int(roi_w * roi_h * 0.0008))
                and bw >= max(18, int(roi_w * 0.03))
                and bh >= max(14, int(roi_h * 0.03))
                and 0.90 <= aspect <= 2.80
                and away_from_border
            )
            if plausible_card:
                hull_area = cv2.contourArea(cv2.convexHull(c))
                if hull_area < roi_w * roi_h * REFERENCE_MIN_ROI_AREA_RATIO:
                    too_small_candidate = True
                else:
                    rejected_candidate = True

    if best_candidate is not None:
        return (*best_candidate, "CARD CANDIDATE") if with_status else best_candidate
    if with_status:
        if too_small_candidate:
            diagnostic = "REFERENCE CANDIDATE TOO SMALL"
        elif rejected_candidate:
            diagnostic = "CARD REJECTED"
        else:
            diagnostic = "NO CARD CANDIDATE"
        return None, None, False, diagnostic
    return None, None, False


def calculate_perspective_transform(card_corners_full):
    """
    Compute planar homography from reference card corners to physical centimeters:
        card_corners_full (pixels in camera frame) -> Physical Coordinates (cm)
    """
    pts = order_corners(card_corners_full)

    top_w = np.linalg.norm(pts[1] - pts[0])
    bot_w = np.linalg.norm(pts[2] - pts[3])
    left_h = np.linalg.norm(pts[3] - pts[0])
    right_h = np.linalg.norm(pts[2] - pts[1])

    avg_w_px = (top_w + bot_w) / 2.0
    avg_h_px = (left_h + right_h) / 2.0

    if avg_w_px <= 10 or avg_h_px <= 10:
        return None

    if avg_w_px >= avg_h_px:
        is_landscape = True
        real_w = REFERENCE_WIDTH_CM   # 8.56 cm
        real_h = REFERENCE_HEIGHT_CM  # 5.398 cm
    else:
        is_landscape = False
        real_w = REFERENCE_HEIGHT_CM  # 5.398 cm
        real_h = REFERENCE_WIDTH_CM   # 8.56 cm

    dst_pts = np.array([
        [0.0, 0.0],
        [real_w, 0.0],
        [real_w, real_h],
        [0.0, real_h]
    ], dtype=np.float32)

    try:
        H_matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    except Exception:
        H_matrix = None

    px_per_cm_w = float(avg_w_px / real_w)
    px_per_cm_h = float(avg_h_px / real_h)

    return H_matrix, px_per_cm_w, px_per_cm_h, is_landscape, (avg_w_px, avg_h_px)


def calibrate_reference(card_corners_roi, roi_x, roi_y, has_approx_quad):
    """
    Calibrate reference card and record its location/signature so it can be
    distinguished from target objects later.
    """
    card_corners_full = card_corners_roi + np.array([roi_x, roi_y], dtype=np.float32)
    res = calculate_perspective_transform(card_corners_full)
    if res is None:
        return None

    H_matrix, px_w, px_h, is_landscape, (avg_w_px, avg_h_px) = res

    # Calculate center of the calibrated card in ROI space
    card_center_roi = (float(np.mean(card_corners_roi[:, 0])), float(np.mean(card_corners_roi[:, 1])))
    card_area_approx = avg_w_px * avg_h_px

    return {
        "H_matrix": H_matrix,
        "px_per_cm_w": px_w,
        "px_per_cm_h": px_h,
        "is_landscape": is_landscape,
        "card_area_px": card_area_approx,
        "card_center_roi": card_center_roi,
        "has_approx_quad": has_approx_quad,
        "corners_full": card_corners_full,
    }


def is_calibrated_reference_active(contour, calibration_data, roi_w, roi_h):
    
    if calibration_data is None:
        return False

    c_area = cv2.contourArea(contour)
    if c_area < 800:
        return False

    bx, by, bw, bh = cv2.boundingRect(contour)
    aspect = max(bw, bh) / max(min(bw, bh), 1)

    cal_area = calibration_data.get("card_area_px", 0)
    cal_center = calibration_data.get("card_center_roi", (0, 0))

    contour_center = (bx + bw / 2.0, by + bh / 2.0)
    dist_from_cal_center = np.hypot(contour_center[0] - cal_center[0], contour_center[1] - cal_center[1])

    # Card is recognized if it is still near its calibrated position (+/- 65px)
    # AND matches the card's area (+/- 30%) and aspect ratio
    if dist_from_cal_center < 65.0:
        area_ratio = c_area / max(cal_area, 1)
        if 0.70 <= area_ratio <= 1.35 and 1.15 <= aspect <= 2.20:
            return True

    return False


def validate_object_quad(points, roi_w, roi_h):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not cv2.isContourConvex(points.astype(np.int32).reshape(-1, 1, 2)):
        return False

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    if x_min <= 4 or y_min <= 4 or x_max >= roi_w - 4 or y_max >= roi_h - 4:
        return False

    sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    if np.min(sides) < 0.08 * np.max(sides):
        return False

    area = cv2.contourArea(points.astype(np.float32))
    return area >= roi_w * roi_h * MIN_TARGET_ROI_AREA_RATIO


def recover_rectangular_contour(contour, edges, roi_w, roi_h):
    """Use contour approximation, then supported near-horizontal/vertical lines."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return contour, False

    bx, by, bw, bh = cv2.boundingRect(contour)
    for epsilon_ratio in (0.02, 0.03, 0.045, 0.06):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) == 4 and validate_object_quad(approx, roi_w, roi_h):
            quad_area = cv2.contourArea(approx)
            contour_area = cv2.contourArea(contour)
            if (quad_area >= 0.55 * bw * bh and
                    contour_area / max(bw * bh, 1) >= 0.60 and
                    quad_area <= 1.45 * max(contour_area, 1)):
                return approx, True

    # Hough recovery is limited to candidates whose contour already spans the
    # proposed object bounds; short internal strokes cannot define four sides.
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if (contour_area / max(bw * bh, 1) < 0.50 or
            contour_area / max(hull_area, 1) < 0.75):
        return contour, False

    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(roi_w, bx + bw), min(roi_h, by + bh)
    crop = edges[y0:y1, x0:x1]
    if crop.size == 0 or min(bw, bh) < 24:
        return contour, False

    min_length = max(18, int(min(bw, bh) * 0.45))
    lines = cv2.HoughLinesP(crop, 1, np.pi / 180, threshold=max(16, min_length // 2),
                            minLineLength=min_length, maxLineGap=max(8, min_length // 4))
    if lines is None:
        return contour, False

    horizontal = []
    vertical = []
    for line in lines.reshape(-1, 4):
        x_start, y_start, x_end, y_end = map(int, line)
        dx = abs(x_end - x_start)
        dy = abs(y_end - y_start)
        if dx >= 2 * dy and dx >= 0.55 * bw:
            horizontal.append((y0 + (y_start + y_end) / 2.0, dx))
        elif dy >= 2 * dx and dy >= 0.55 * bh:
            vertical.append((x0 + (x_start + x_end) / 2.0, dy))

    if len(horizontal) < 2 or len(vertical) < 2:
        return contour, False

    top = min(y for y, _ in horizontal)
    bottom = max(y for y, _ in horizontal)
    left = min(x for x, _ in vertical)
    right = max(x for x, _ in vertical)
    if bottom - top < 0.55 * bh or right - left < 0.55 * bw:
        return contour, False

    quad = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    if not validate_object_quad(quad, roi_w, roi_h):
        return contour, False
    quad_area = cv2.contourArea(quad)
    if not 0.55 * bw * bh <= quad_area <= 1.35 * bw * bh:
        return contour, False
    return quad.reshape(-1, 1, 2).astype(np.int32), True


def score_candidate_contour(contour, roi_w, roi_h, prev_bbox=None, edges=None):
   
    area = cv2.contourArea(contour)
    roi_area = roi_w * roi_h
    bx, by, bw, bh = cv2.boundingRect(contour)
    # Edge candidates are reported before size rejection to prevent clipped measurements.
    touching_edge = (bx <= 4 or by <= 4 or (bx + bw) >= (roi_w - 4) or (by + bh) >= (roi_h - 4))
    if touching_edge:
        return 0.0, True

    min_area = max(MIN_CONTOUR_AREA, int(roi_area * MIN_TARGET_ROI_AREA_RATIO))
    thin_metrics = _thin_object_metrics(contour, roi_w, roi_h, edges)
    is_thin_object = thin_metrics is not None
    if not is_thin_object and (area < min_area or (bw * bh) < roi_area * MIN_TARGET_BBOX_ROI_RATIO):
        return 0.0, False

    # 2. Reject extreme thin lines (creases, glare lines)
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect > 9.0 and min(bw, bh) < 12 and not is_thin_object:
        return 0.0, False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = min(1.0, area / max(hull_area, 1))

    # Reject fragmented spaghetti edges
    if solidity < (0.35 if is_thin_object else 0.22):
        return 0.0, False

    extent = min(1.0, area / max(bw * bh, 1))
    area_norm = min(1.0, area / (roi_area * 0.30))
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    polygon_support = 1.0 if len(approx) == 4 and cv2.isContourConvex(approx) else 0.0
    completeness = min(1.0, 0.55 * solidity + 0.30 * extent + 0.15 * polygon_support)

    # Stability score based on overlap with previous frame
    stability = 0.5
    if prev_bbox is not None:
        px, py, pw, ph = prev_bbox
        ix1 = max(bx, px)
        iy1 = max(by, py)
        ix2 = min(bx + bw, px + pw)
        iy2 = min(by + bh, py + ph)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        union = (bw * bh) + (pw * ph) - intersection
        if union > 0:
            stability = intersection / union

    score = (
        (area_norm * 0.40)
        + (completeness * 0.25)
        + (solidity * 0.15)
        + (extent * 0.10)
        + (stability * 0.10)
    )

    # Thin objects occupy little area, so reward their measurable length and
    # coherent filled silhouette while retaining the same temporal confirmation.
    if is_thin_object:
        length, thickness, support = thin_metrics
        length_score = min(1.0, length / max(roi_w, roi_h) * 3.0)
        score = (0.30 * length_score + 0.25 * solidity + 0.20 * extent
                 + 0.15 * support + 0.10 * stability)

    return float(score), False


def _thin_object_metrics(contour, roi_w, roi_h, edges=None):
    
    if contour is None or len(contour) < 4:
        return None
    rect = cv2.minAreaRect(contour)
    long_side, short_side = sorted((float(rect[1][0]), float(rect[1][1])), reverse=True)
    min_length = max(90.0, min(roi_w, roi_h) * 0.18)
    if long_side < min_length or short_side < 4.0 or short_side > 28.0:
        return None
    area = cv2.contourArea(contour)
    rotated_extent = area / max(long_side * short_side, 1.0)
    solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    if area < 250 or rotated_extent < 0.55 or solidity < 0.45:
        return None

    support = 1.0
    if edges is not None:
        # Sample the contour itself against a slightly tolerant edge map. This
        # works for rotated objects and avoids treating a single isolated line as
        # a complete object boundary.
        contour_mask = np.zeros(edges.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, 1)
        perimeter_pixels = cv2.countNonZero(contour_mask)
        if perimeter_pixels:
            nearby_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            support = cv2.countNonZero(cv2.bitwise_and(contour_mask, nearby_edges)) / perimeter_pixels
        if support < 0.35:
            return None
    return long_side, short_side, float(support)


def _candidate_encloses(outer_contour, inner_contour):
    """Return true only when bbox, contour position, and relative size agree."""
    ox, oy, ow, oh = cv2.boundingRect(outer_contour)
    ix, iy, iw, ih = cv2.boundingRect(inner_contour)
    outer_bbox_area = max(ow * oh, 1)
    inner_bbox_area = max(iw * ih, 1)
    intersection_w = max(0, min(ox + ow, ix + iw) - max(ox, ix))
    intersection_h = max(0, min(oy + oh, iy + ih) - max(oy, iy))
    inner_bbox_coverage = (intersection_w * intersection_h) / inner_bbox_area
    bbox_area_ratio = outer_bbox_area / inner_bbox_area
    contour_area_ratio = cv2.contourArea(outer_contour) / max(cv2.contourArea(inner_contour), 1.0)

    inner_center = (ix + iw / 2.0, iy + ih / 2.0)
    center_inside = cv2.pointPolygonTest(
        outer_contour, (float(inner_center[0]), float(inner_center[1])), False
    ) >= 0
    inner_points = inner_contour.reshape(-1, 2)
    if len(inner_points) > 16:
        sample_indices = np.linspace(0, len(inner_points) - 1, 16, dtype=int)
        inner_points = inner_points[sample_indices]
    points_inside = sum(
        cv2.pointPolygonTest(outer_contour, (float(point[0]), float(point[1])), False) >= 0
        for point in inner_points
    )
    contour_coverage = points_inside / max(len(inner_points), 1)

    outer_area = cv2.contourArea(outer_contour)
    outer_hull_area = cv2.contourArea(cv2.convexHull(outer_contour))
    outer_extent = outer_area / outer_bbox_area
    outer_solidity = outer_area / max(outer_hull_area, 1.0)
    outer_aspect = max(ow, oh) / max(min(ow, oh), 1)
    coherent_outer = outer_extent >= 0.28 and outer_solidity >= 0.42 and outer_aspect <= 8.0

    return (
        coherent_outer
        and inner_bbox_coverage >= 0.90
        and center_inside
        and contour_coverage >= 0.70
        and bbox_area_ratio >= 2.5
        and contour_area_ratio >= 1.8
    )


def select_primary_target(top_level_contours, calibration_data, roi_w, roi_h, target_tracker,
                          edges=None, debug_records=None):
    
    ref_contour = None
    target_candidates = []
    candidate_record_ids = {}
    if debug_records is not None:
        debug_records.clear()
        for contour_id, contour in enumerate(top_level_contours):
            debug_records.append({
                "id": contour_id,
                "contour": contour,
                "classification": "RED",
                "reason": "not eligible",
                "score": 0.0,
            })

    # Exclude the calibrated reference before deriving target-relative thresholds.
    remaining_contours = []
    for contour_id, contour in enumerate(top_level_contours):
        if is_calibrated_reference_active(contour, calibration_data, roi_w, roi_h):
            ref_contour = contour
            if debug_records is not None:
                debug_records[contour_id]["reason"] = "calibrated reference excluded"
        else:
            remaining_contours.append((contour_id, contour))

    # Relative size threshold rejects details even when they form closed contours.
    max_candidate_area = 0.0
    for _, c in remaining_contours:
        a = cv2.contourArea(c)
        if a > max_candidate_area:
            max_candidate_area = a

    relative_min_area = max(MIN_CONTOUR_AREA, roi_w * roi_h * MIN_TARGET_ROI_AREA_RATIO,
                            max_candidate_area * 0.18)
    had_partial_candidate = False

    for contour_id, c in remaining_contours:
        record = debug_records[contour_id] if debug_records is not None else None
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        touching_edge = (bx <= 4 or by <= 4 or
                 (bx + bw) >= roi_w - 4 or (by + bh) >= roi_h - 4)
        if touching_edge:
            target_candidates.append((c, 0.0, True))
            if record is not None:
                record["classification"] = "BLUE"
                record["reason"] = "Focus Zone boundary"
            continue

        is_thin_object = _thin_object_metrics(c, roi_w, roi_h, edges) is not None
        if (not is_thin_object and (area < relative_min_area or
                bw * bh < roi_w * roi_h * MIN_TARGET_BBOX_ROI_RATIO)):
            had_partial_candidate = had_partial_candidate or area >= MIN_CONTOUR_AREA
            if record is not None:
                record["reason"] = "relative area/bbox filter"
            continue

        candidate = c
        if edges is not None:
            candidate, _ = recover_rectangular_contour(c, edges, roi_w, roi_h)
        score, touching_edge = score_candidate_contour(candidate, roi_w, roi_h,
                                                       prev_bbox=target_tracker.current_bbox,
                                                       edges=edges)
        if score == 0.0 and not touching_edge and area >= MIN_CONTOUR_AREA:
            had_partial_candidate = True
        target_candidates.append((candidate, score, touching_edge))
        candidate_record_ids[id(candidate)] = contour_id
        if record is not None:
            record["candidate_contour"] = candidate
            record["score"] = score
            record["classification"] = "GREEN" if score > 0.15 else "RED"
            record["reason"] = "accepted outer candidate" if score > 0.15 else "candidate score rejected"

    # Remove sticker/text/screen candidates enclosed by a coherent larger object.
    eligible = [candidate for candidate in target_candidates
                if candidate[1] > 0.15 and not candidate[2]]
    internal_ids = set()
    enclosure_counts = {}
    for inner_index, inner in enumerate(eligible):
        for outer_index, outer in enumerate(eligible):
            if inner_index == outer_index:
                continue
            if _candidate_encloses(outer[0], inner[0]):
                internal_ids.add(id(inner))
                enclosure_counts[outer_index] = enclosure_counts.get(outer_index, 0) + 1

    filtered_candidates = []
    for candidate in target_candidates:
        if id(candidate) in internal_ids:
            had_partial_candidate = True
            record_id = candidate_record_ids.get(id(candidate[0]))
            if debug_records is not None and record_id is not None:
                debug_records[record_id]["classification"] = "YELLOW"
                debug_records[record_id]["reason"] = "contained inside larger coherent candidate"
            continue
        filtered_candidates.append(candidate)

    # A coherent outer boundary enclosing details gets priority over clean details.
    enclosure_bonus = {}
    for outer_index, enclosed_count in enclosure_counts.items():
        outer_contour = eligible[outer_index][0]
        enclosure_bonus[id(outer_contour)] = min(0.24, 0.12 * enclosed_count)
    filtered_candidates = [
        (contour, min(1.0, score + enclosure_bonus.get(id(contour), 0.0)), touching_edge)
        for contour, score, touching_edge in filtered_candidates
    ]

    # Update tracker to maintain temporal stability
    target_contour, is_touching_edge, status_type = target_tracker.update(
        filtered_candidates, roi_w, roi_h
    )

    if debug_records is not None:
        selected_id = None
        if target_contour is not None:
            selected_bbox = cv2.boundingRect(target_contour)
            best_iou = 0.0
            for record in debug_records:
                candidate = record.get("candidate_contour", record["contour"])
                x, y, w, h = cv2.boundingRect(candidate)
                sx, sy, sw, sh = selected_bbox
                intersection = max(0, min(x + w, sx + sw) - max(x, sx)) * max(
                    0, min(y + h, sy + sh) - max(y, sy)
                )
                union = w * h + sw * sh - intersection
                iou = intersection / max(union, 1)
                if iou > best_iou:
                    best_iou = iou
                    selected_id = record["id"]
            if selected_id is not None and best_iou >= 0.50:
                debug_records[selected_id]["reason"] = "SELECTED TARGET"
        for record in debug_records:
            if record["classification"] == "BLUE":
                continue
            if record["reason"] == "relative area/bbox filter":
                x, y, w, h = cv2.boundingRect(record["contour"])
                record["classification"] = "BLUE" if (
                    x <= 4 or y <= 4 or x + w >= roi_w - 4 or y + h >= roi_h - 4
                ) else "RED"
        debug_records.append({"selected_id": selected_id})

    if target_contour is None and status_type in ("EMPTY", "SEARCHING") and had_partial_candidate:
        status_type = "UNCLEAR"

    return target_contour, ref_contour, is_touching_edge, status_type


def save_target_detection_debug(roi, debug_images, contours, records, target_contour):
    """Write one requested diagnostic snapshot; never called from the frame loop automatically."""
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "01_roi.png"), roi)
    cv2.imwrite(os.path.join(debug_dir, "02_gray.png"), debug_images["gray"])
    cv2.imwrite(os.path.join(debug_dir, "03_blur.png"), debug_images["blur"])
    cv2.imwrite(os.path.join(debug_dir, "04_canny.png"), debug_images["canny"])
    cv2.imwrite(os.path.join(debug_dir, "05_morphology.png"), debug_images["morphology"])
    if "segmentation_mask" in debug_images:
        cv2.imwrite(os.path.join(debug_dir, "08_segmentation_mask.png"),
                    debug_images["segmentation_mask"])

    contour_view = cv2.cvtColor(debug_images["morphology"], cv2.COLOR_GRAY2BGR)
    candidate_view = roi.copy()
    selected_id = next((record.get("selected_id") for record in records
                        if "selected_id" in record), None)
    visible_records = [record for record in records if "id" in record]

    accepted_records = [record for record in visible_records
                        if record["classification"] == "GREEN"]
    for record in visible_records:
        if record["classification"] != "RED":
            continue
        inner_contour = record["contour"]
        if any(
            outer["id"] != record["id"] and
            _candidate_encloses(outer.get("candidate_contour", outer["contour"]), inner_contour)
            for outer in accepted_records
        ):
            record["classification"] = "YELLOW"
            record["reason"] = "geometrically inside accepted outer candidate"

    for record in visible_records:
        contour_id = record["id"]
        contour = record["contour"]
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / max(hull_area, 1.0)
        extent = area / max(width * height, 1)
        aspect = max(width, height) / max(min(width, height), 1)
        print(
            f"C{contour_id}: bbox=({x},{y},{width},{height}), area={area:.1f}, "
            f"width={width}, height={height}, aspect={aspect:.3f}, "
            f"solidity={solidity:.3f}, extent={extent:.3f}, parent=N/A (RETR_EXTERNAL), "
            f"class={record['classification']}, reason={record['reason']}"
        )

        cv2.drawContours(contour_view, [record["contour"]], -1, (0, 255, 0), 2)
        cv2.putText(contour_view, f"C{contour_id}", (x, max(16, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        class_color = {
            "GREEN": (0, 220, 0),
            "RED": (0, 0, 255),
            "YELLOW": (0, 220, 255),
            "BLUE": (255, 100, 0),
        }.get(record["classification"], (0, 0, 255))
        cv2.drawContours(candidate_view, [record["contour"]], -1, class_color, 2)
        cv2.rectangle(candidate_view, (x, y), (x + width, y + height), class_color, 1)
        label = f"C{contour_id} {record['classification']}"
        cv2.putText(candidate_view, label, (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, class_color, 2, cv2.LINE_AA)

    if target_contour is not None:
        print(f"SELECTED TARGET = C{selected_id}" if selected_id is not None
              else "SELECTED TARGET = held contour (not present in current candidates)")
    else:
        print("SELECTED TARGET = NONE")
    cv2.putText(candidate_view, f"SELECTED TARGET = C{selected_id}" if selected_id is not None
                else "SELECTED TARGET = NONE", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(debug_dir, "06_contours.png"), contour_view)
    cv2.imwrite(os.path.join(debug_dir, "07_candidates.png"), candidate_view)
    print(f"[DEBUG] One target-detection snapshot saved to: {debug_dir}")


def measure_target_object(target_contour, calibration_data, roi_x, roi_y):
    
    if target_contour is None or len(target_contour) < 3:
        return None

    min_rect = cv2.minAreaRect(target_contour)
    box_pts_roi = cv2.boxPoints(min_rect)
    box_pts_full = box_pts_roi + np.array([roi_x, roi_y], dtype=np.float32)

    dim1_px, dim2_px = min_rect[1]
    w_px = int(round(max(dim1_px, dim2_px)))
    h_px = int(round(min(dim1_px, dim2_px)))

    w_cm = 0.0
    h_cm = 0.0
    used_perspective = False

    if calibration_data is not None:
        H_matrix = calibration_data.get("H_matrix", None)

        if H_matrix is not None:
            try:
                pts_input = box_pts_full.reshape(-1, 1, 2)
                pts_cm = cv2.perspectiveTransform(pts_input, H_matrix).squeeze()

                d1 = float(np.linalg.norm(pts_cm[1] - pts_cm[0]))
                d2 = float(np.linalg.norm(pts_cm[2] - pts_cm[1]))

                if 0.1 <= d1 <= 200.0 and 0.1 <= d2 <= 200.0:
                    w_cm = max(d1, d2)
                    h_cm = min(d1, d2)
                    used_perspective = True
            except Exception:
                used_perspective = False

        if not used_perspective:
            px_w = calibration_data.get("px_per_cm_w", 50.0)
            px_h = calibration_data.get("px_per_cm_h", 50.0)
            avg_scale = (px_w + px_h) / 2.0
            if avg_scale > 0:
                w_cm = w_px / avg_scale
                h_cm = h_px / avg_scale

    return {
        "w_cm": w_cm,
        "h_cm": h_cm,
        "w_px": w_px,
        "h_px": h_px,
        "box_pts_full": box_pts_full.astype(np.int32),
        "used_perspective": used_perspective,
    }


def calculate_measurement_quality(calibration_data, target_data, stabilizer):
    if calibration_data is None or target_data is None:
        return "LOW"

    score = 0
    if target_data.get("used_perspective", False):
        score += 2
    else:
        score += 1

    if calibration_data.get("has_approx_quad", False):
        score += 1

    w_px = target_data.get("w_px", 0)
    h_px = target_data.get("h_px", 0)
    if w_px >= 120 and h_px >= 80:
        score += 2
    elif w_px >= 60 and h_px >= 40:
        score += 1

    var = stabilizer.get_variance()
    if var < 0.35:
        score += 2
    elif var < 1.20:
        score += 1

    if score >= 5:
        return "GOOD"
    elif score >= 3:
        return "FAIR"
    else:
        return "LOW"


def draw_title(canvas, sw, header_h):
    cv2.rectangle(canvas, (0, 0), (sw, header_h), (18, 20, 26), -1)
    cv2.line(canvas, (0, header_h), (sw, header_h), (45, 50, 65), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    title_text = "VISIONRULER"
    t_scale = 0.85
    t_size = cv2.getTextSize(title_text, font, t_scale, 2)[0]
    t_x = (sw - t_size[0]) // 2
    cv2.putText(canvas, title_text, (t_x, 32), font, t_scale, (255, 255, 255), 2, cv2.LINE_AA)

    sub_text = "REAL-TIME OBJECT MEASUREMENT"
    s_scale = 0.42
    s_size = cv2.getTextSize(sub_text, font, s_scale, 1)[0]
    s_x = (sw - s_size[0]) // 2
    cv2.putText(canvas, sub_text, (s_x, 52), font, s_scale, (160, 185, 210), 1, cv2.LINE_AA)


def draw_focus_zone(cam_img, roi_scaled, status_type):
    """
    Renders Focus Zone boundary based on state:
    - TARGET_DETECTED: Vibrant Green
    - EDGE_WARNING: Amber/Red
    - EMPTY: Bright Yellow
    """
    rx, ry, rw, rh = roi_scaled

    if status_type == "TARGET_DETECTED":
        border_col = (0, 230, 100)
        status_txt = "OBJECT DETECTED"
    elif status_type == "TARGET_LOCKED":
        border_col = (0, 230, 100)     # Green
        status_txt = "TARGET LOCKED"
    elif status_type == "TARGET_TRACKING":
        border_col = (0, 230, 100)
        status_txt = "TARGET TRACKING"
    elif status_type == "TARGET_TRACKING_HOLDING":
        border_col = (0, 165, 255)
        status_txt = "TARGET TRACKING - HOLDING"
    elif status_type.startswith("CONFIRMING"):
        border_col = (0, 215, 255)
        status_txt = f"TARGET: {status_type}"
    elif status_type == "EDGE_WARNING":
        border_col = (0, 100, 255)     # Amber/Red
        status_txt = "OBJECT TOO CLOSE TO EDGE"
    elif status_type == "UNCLEAR":
        border_col = (0, 165, 255)
        status_txt = "OBJECT BOUNDARY UNCLEAR"
    else:
        border_col = (0, 215, 255)     # Yellow
        status_txt = "TARGET: SEARCHING"

    cv2.rectangle(cam_img, (rx, ry), (rx + rw, ry + rh), border_col, 2)

    c_len = 26
    c_thick = 3
    # Corner brackets
    cv2.line(cam_img, (rx, ry), (rx + c_len, ry), border_col, c_thick)
    cv2.line(cam_img, (rx, ry), (rx, ry + c_len), border_col, c_thick)
    cv2.line(cam_img, (rx + rw, ry), (rx + rw - c_len, ry), border_col, c_thick)
    cv2.line(cam_img, (rx + rw, ry), (rx + rw, ry + c_len), border_col, c_thick)
    cv2.line(cam_img, (rx, ry + rh), (rx + c_len, ry + rh), border_col, c_thick)
    cv2.line(cam_img, (rx, ry + rh), (rx, ry + rh - c_len), border_col, c_thick)
    cv2.line(cam_img, (rx + rw, ry + rh), (rx + rw - c_len, ry + rh), border_col, c_thick)
    cv2.line(cam_img, (rx + rw, ry + rh), (rx + rw, ry + rh - c_len), border_col, c_thick)

    font = cv2.FONT_HERSHEY_SIMPLEX
    b_scale = 0.48
    b_size = cv2.getTextSize(status_txt, font, b_scale, 2)[0]
    b_x = rx + (rw - b_size[0]) // 2
    b_y = ry - 12 if ry - 12 > 20 else ry + 24

    cv2.rectangle(cam_img, (b_x - 8, b_y - b_size[1] - 4),
                  (b_x + b_size[0] + 8, b_y + 4), (15, 18, 24), -1)
    cv2.rectangle(cam_img, (b_x - 8, b_y - b_size[1] - 4),
                  (b_x + b_size[0] + 8, b_y + 4), border_col, 1)
    cv2.putText(cam_img, status_txt, (b_x, b_y), font, b_scale, border_col, 2, cv2.LINE_AA)


def draw_measurement_info(cam_img, ref_corners_full, target_data, calibration_data, sx, sy):
    """Draw rotated bounding boxes and measurement tags on camera feed."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Reference Card in Orange
    if ref_corners_full is not None:
        scaled_corners = (ref_corners_full * np.array([sx, sy])).astype(np.int32)
        cv2.polylines(cam_img, [scaled_corners], isClosed=True, color=(0, 165, 255), thickness=2)

        lbl = "REFERENCE: STANDARD CARD"
        f_size = cv2.getTextSize(lbl, font, 0.44, 1)[0]
        lx = max(scaled_corners[0][0], 10)
        ly = max(scaled_corners[0][1] - 8, 20)

        cv2.rectangle(cam_img, (lx - 4, ly - f_size[1] - 3), (lx + f_size[0] + 4, ly + 3), (15, 18, 24), -1)
        cv2.rectangle(cam_img, (lx - 4, ly - f_size[1] - 3), (lx + f_size[0] + 4, ly + 3), (0, 165, 255), 1)
        cv2.putText(cam_img, lbl, (lx, ly), font, 0.44, (0, 165, 255), 1, cv2.LINE_AA)

    # 2. Target Outer Silhouette in Cyan/Green
    if target_data is not None:
        box_pts = target_data["box_pts_full"]
        scaled_box = (box_pts * np.array([sx, sy])).astype(np.int32)
        target_col = (0, 240, 220)

        cv2.polylines(cam_img, [scaled_box], isClosed=True, color=target_col, thickness=2)

        is_cal = (calibration_data is not None)
        w_cm = target_data.get("w_cm", 0.0)
        h_cm = target_data.get("h_cm", 0.0)
        w_px = target_data.get("w_px", 0)
        h_px = target_data.get("h_px", 0)

        if is_cal and w_cm > 0:
            label_text = f"WIDTH: {w_cm:.2f} cm (est)  |  HEIGHT: {h_cm:.2f} cm (est)"
        else:
            label_text = f"WIDTH: {w_px} px  |  HEIGHT: {h_px} px"

        f_size = cv2.getTextSize(label_text, font, 0.48, 1)[0]
        top_pt = min(scaled_box, key=lambda p: p[1])
        lbl_x = max(int(top_pt[0] - f_size[0] // 2), 10)
        lbl_y = max(int(top_pt[1] - 10), 20)

        cv2.rectangle(cam_img, (lbl_x - 4, lbl_y - f_size[1] - 3), (lbl_x + f_size[0] + 4, lbl_y + 3), (15, 18, 24), -1)
        cv2.rectangle(cam_img, (lbl_x - 4, lbl_y - f_size[1] - 3), (lbl_x + f_size[0] + 4, lbl_y + 3), target_col, 1)
        cv2.putText(cam_img, label_text, (lbl_x, lbl_y), font, 0.48, target_col, 1, cv2.LINE_AA)


def draw_measurement_panel(canvas, rect, calibration_data, target_data, quality_str,
                           cal_status_msg="", is_edge_warning=False, status_type="EMPTY"):
    """Draw Measurement Status panel in lower-left dashboard."""
    px, py, pw, ph = rect

    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (22, 24, 32), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (45, 50, 65), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Card Title
    cv2.putText(canvas, "MEASUREMENT STATUS", (px + 18, py + 25),
                font, 0.50, (200, 220, 240), 1, cv2.LINE_AA)
    cv2.line(canvas, (px + 18, py + 33), (px + pw - 18, py + 33), (40, 45, 58), 1)

    is_calibrated = (calibration_data is not None)

    # Line 1: Calibration Status & Reference Card Indicator
    if is_calibrated:
        cal_str = "CALIBRATION: READY"
        cal_col = (0, 240, 120)
        ref_badge = "REFERENCE: CALIBRATED"
        ref_col = (0, 165, 255)
    else:
        cal_str = "CALIBRATION: NOT READY"
        cal_col = (60, 80, 255)
        ref_badge = "REFERENCE: NOT CALIBRATED"
        ref_col = (130, 140, 155)

    cv2.putText(canvas, cal_str, (px + 18, py + 56), font, 0.50, cal_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, ref_badge, (px + pw - 250, py + 56), font, 0.44, ref_col, 1, cv2.LINE_AA)

    # Line 2: Scale info or Calibration Guidance
    if cal_status_msg.startswith("CALIBRATION:") and cal_status_msg != "CALIBRATION: READY":
        cv2.putText(canvas, cal_status_msg, (px + 18, py + 78), font, 0.44, (0, 215, 255), 1, cv2.LINE_AA)
    elif is_calibrated:
        px_w = calibration_data.get("px_per_cm_w", 50.0)
        px_h = calibration_data.get("px_per_cm_h", 50.0)
        avg_scale = (px_w + px_h) / 2.0
        persp_mode = "PERSPECTIVE: ACTIVE" if calibration_data.get("H_matrix") is not None else "PERSPECTIVE: OFF"
        scale_txt = f"SCALE: {avg_scale:.1f} px/cm  |  {persp_mode}"
        cv2.putText(canvas, scale_txt, (px + 18, py + 78), font, 0.44, (180, 190, 205), 1, cv2.LINE_AA)
    elif cal_status_msg:
        cv2.putText(canvas, cal_status_msg, (px + 18, py + 78), font, 0.44, (0, 215, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Place standard card inside the box & press C to calibrate",
                    (px + 18, py + 78), font, 0.44, (180, 190, 205), 1, cv2.LINE_AA)

    # Line 3: Target Status Indicator
    has_target = (target_data is not None)
    if is_edge_warning:
        target_badge = "STATUS: OBJECT TOO CLOSE TO EDGE"
        target_b_col = (0, 100, 255)
    elif status_type == "UNCLEAR":
        target_badge = "STATUS: OBJECT BOUNDARY UNCLEAR"
        target_b_col = (0, 165, 255)
    elif has_target and not is_calibrated:
        target_badge = "TARGET: DETECTED"
        target_b_col = (0, 240, 120)
    elif status_type.startswith("CONFIRMING"):
        target_badge = f"TARGET: {status_type}"
        target_b_col = (0, 215, 255)
    elif status_type == "TARGET_LOCKED":
        target_badge = "TARGET: LOCKED"
        target_b_col = (0, 240, 120)
    elif status_type == "TARGET_TRACKING":
        target_badge = "TARGET: TRACKING"
        target_b_col = (0, 240, 120)
    elif status_type == "TARGET_TRACKING_HOLDING":
        target_badge = "TARGET: TRACKING - HOLDING"
        target_b_col = (0, 165, 255)
    elif has_target:
        target_badge = "TARGET: TRACKING"
        target_b_col = (0, 240, 120)
    else:
        target_badge = "TARGET: SEARCHING"
        target_b_col = (130, 140, 155)

    cv2.putText(canvas, target_badge, (px + 18, py + 98), font, 0.46, target_b_col, 1, cv2.LINE_AA)
    cv2.line(canvas, (px + 18, py + 108), (px + pw - 18, py + 108), (40, 45, 58), 1)

    # Measurement Readout
    if is_edge_warning:
        cv2.putText(canvas, "OBJECT TOO CLOSE TO EDGE", (px + 18, py + 145), font, 0.54, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Move object fully inside the box to measure", (px + 18, py + 172), font, 0.44, (180, 190, 205), 1, cv2.LINE_AA)
    elif status_type == "UNCLEAR":
        cv2.putText(canvas, "OBJECT BOUNDARY UNCLEAR", (px + 18, py + 145), font, 0.54, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Adjust object so its complete outer edge is visible", (px + 18, py + 172), font, 0.44, (180, 190, 205), 1, cv2.LINE_AA)
    elif status_type.startswith("CONFIRMING"):
        cv2.putText(canvas, "CONFIRMING OBJECT BOUNDARY", (px + 18, py + 145), font, 0.54, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Hold the object steady inside the Focus Zone", (px + 18, py + 172), font, 0.44, (180, 190, 205), 1, cv2.LINE_AA)
    elif has_target:
        w_cm = target_data.get("w_cm", 0.0)
        h_cm = target_data.get("h_cm", 0.0)
        w_px = target_data.get("w_px", 0)
        h_px = target_data.get("h_px", 0)

        if is_calibrated and w_cm > 0:
            w_str = f"WIDTH  : {w_cm:6.2f} cm  (estimated)"
            h_str = f"HEIGHT : {h_cm:6.2f} cm  (estimated)"
            px_str = f"Bounding: {w_px} px x {h_px} px (rotated)"

            cv2.putText(canvas, w_str, (px + 18, py + 136), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, h_str, (px + 18, py + 164), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, px_str, (px + 18, py + 188), font, 0.42, (130, 145, 160), 1, cv2.LINE_AA)

            q_col = (0, 240, 120) if quality_str == "GOOD" else ((0, 215, 255) if quality_str == "FAIR" else (60, 80, 255))
            q_str = f"MEASUREMENT QUALITY: {quality_str}"
            cv2.putText(canvas, q_str, (px + pw - 275, py + 188), font, 0.44, q_col, 1, cv2.LINE_AA)
        else:
            w_str = f"WIDTH  : {w_px} px"
            h_str = f"HEIGHT : {h_px} px"
            note_str = "Press C to calibrate with standard card to view centimeters"

            cv2.putText(canvas, w_str, (px + 18, py + 136), font, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, h_str, (px + 18, py + 162), font, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, note_str, (px + 18, py + 188), font, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "NO TARGET OBJECT DETECTED", (px + 18, py + 145), font, 0.54, (140, 150, 165), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Place an object inside the focus box to measure", (px + 18, py + 172), font, 0.44, (110, 120, 135), 1, cv2.LINE_AA)


def draw_canny_panel(canvas, rect, edges):
    px, py, pw, ph = rect

    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (22, 24, 32), -1)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (45, 50, 65), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "CANNY EDGES", (px + 18, py + 24), font, 0.50, (200, 220, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "FOCUS ZONE", (px + 18, py + 38), font, 0.38, (140, 160, 180), 1, cv2.LINE_AA)
    cv2.line(canvas, (px + 18, py + 44), (px + pw - 18, py + 44), (40, 45, 58), 1)

    prev_x = px + 18
    prev_y = py + 52
    prev_w = pw - 36
    prev_h = ph - 64

    if prev_w > 10 and prev_h > 10:
        eh, ew = edges.shape[:2]
        edge_aspect = ew / max(eh, 1)
        target_aspect = prev_w / prev_h

        if edge_aspect > target_aspect:
            fit_w = prev_w
            fit_h = int(prev_w / edge_aspect)
        else:
            fit_h = prev_h
            fit_w = int(prev_h * edge_aspect)

        offset_x = prev_x + (prev_w - fit_w) // 2
        offset_y = prev_y + (prev_h - fit_h) // 2

        scaled_edges = cv2.resize(edges, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)
        scaled_edges_bgr = cv2.cvtColor(scaled_edges, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(canvas, (offset_x - 1, offset_y - 1),
                      (offset_x + fit_w + 1, offset_y + fit_h + 1), (50, 55, 70), 1)
        canvas[offset_y:offset_y + fit_h, offset_x:offset_x + fit_w] = scaled_edges_bgr


def draw_instructions(canvas, sw, sh, footer_h):
    fy = sh - footer_h
    cv2.rectangle(canvas, (0, fy), (sw, sh), (18, 20, 26), -1)
    cv2.line(canvas, (0, fy), (sw, fy), (45, 50, 65), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    control_text = "C  Calibrate      |      R  Reset      |      Q  Exit"
    scale = 0.44
    thick = 1
    t_size = cv2.getTextSize(control_text, font, scale, thick)[0]
    t_x = (sw - t_size[0]) // 2
    t_y = fy + (footer_h + t_size[1]) // 2

    cv2.putText(canvas, control_text, (t_x, t_y), font, scale, (210, 225, 240), thick, cv2.LINE_AA)


def build_fullscreen_display(frame, edges, roi_coords, ref_corners_full, target_data,
                             calibration_data, quality_str, cal_status_msg, status_type,
                             is_edge_warning, sw, sh):
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

    header_h = 60
    footer_h = 42
    lower_h = 215

    panel_y = sh - footer_h - lower_h - 10
    total_panel_w = int(sw * 0.90)
    gap = 20
    left_w = int(total_panel_w * 0.60)
    right_w = total_panel_w - left_w - gap
    left_x = (sw - total_panel_w) // 2
    right_x = left_x + left_w + gap

    cam_area_y = header_h + 10
    cam_area_h = panel_y - cam_area_y - 12

    orig_h, orig_w = frame.shape[:2]
    cam_aspect = orig_w / max(orig_h, 1)

    fit_h = cam_area_h
    fit_w = int(fit_h * cam_aspect)
    max_cam_w = int(sw * 0.90)
    if fit_w > max_cam_w:
        fit_w = max_cam_w
        fit_h = int(fit_w / cam_aspect)

    cam_x = (sw - fit_w) // 2
    cam_y = cam_area_y + (cam_area_h - fit_h) // 2

    cam_display = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)
    sx = fit_w / orig_w
    sy = fit_h / orig_h

    roi_x, roi_y, roi_w, roi_h = roi_coords
    roi_scaled = (int(roi_x * sx), int(roi_y * sy), int(roi_w * sx), int(roi_h * sy))

    draw_focus_zone(cam_display, roi_scaled, status_type)
    draw_measurement_info(cam_display, ref_corners_full, target_data, calibration_data, sx, sy)

    cv2.rectangle(canvas, (cam_x - 1, cam_y - 1), (cam_x + fit_w + 1, cam_y + fit_h + 1), (45, 50, 65), 1)
    canvas[cam_y:cam_y + fit_h, cam_x:cam_x + fit_w] = cam_display

    draw_title(canvas, sw, header_h)
    draw_measurement_panel(canvas, (left_x, panel_y, left_w, lower_h),
                           calibration_data, target_data, quality_str, cal_status_msg,
                           is_edge_warning, status_type)
    draw_canny_panel(canvas, (right_x, panel_y, right_w, lower_h), edges)
    draw_instructions(canvas, sw, sh, footer_h)

    return canvas


def main():
    screen_w, screen_h = get_screen_resolution()
    print(f"[OK] Detected Screen Resolution: {screen_w} x {screen_h}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    print("[OK] Webcam started successfully!")
    print("=" * 65)
    print("VisionRuler - Complete Outer Object Detection System")
    print("=" * 65)
    print()
    print("WORKFLOW:")
    print("  1. Place standard bank card inside the central FOCUS BOX.")
    print("  2. Press [C] to calibrate (samples stable frames with outlier rejection).")
    print("  3. Keep the card or remove it from the scene.")
    print("  4. Place any object (phone, book, bottle, curved, or rotated) inside.")
    print("  5. The entire outer silhouette is measured; internal details are ignored.")
    print()
    print("CONTROLS:")
    print("  [C]   Calibrate reference card")
    print("  [R]   Reset calibration")
    print("  [D]   Save one target-detection diagnostic frame")
    print("  [Q]   Exit application")
    print()

    # Configure Fullscreen Window
    window_name = "VisionRuler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calibration_data = None
    target_tracker = TargetTracker()
    stabilizer = MeasurementStabilizer(history_len=7)

    # Multi-frame calibration state
    cal_state = {
        "active": False,
        "samples": [],
        "frames_waited": 0,
        "message": "",
        "diagnostic": "CALIBRATION: SEARCHING CARD"
    }
    frame_index = 0
    last_segmentation_frame = 1 - SEGMENTATION_INTERVAL
    last_segmentation_mask = None
    last_segmentation_contour = None
    previous_roi_gray = None
    capture_debug_next_frame = False
    perf_samples = []

    while True:
        frame_start = time.perf_counter()
        capture_start = time.perf_counter()
        success, frame = camera.read()
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
        if not success:
            print("[ERROR] Could not read frame.")
            break
        frame_index += 1

        # 1. Extract Focus Zone ROI
        roi, roi_coords = get_focus_roi(frame, ROI_WIDTH_RATIO, ROI_HEIGHT_RATIO)
        roi_x, roi_y, roi_w, roi_h = roi_coords
        current_roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        debug_capture_now = capture_debug_next_frame
        capture_debug_next_frame = False
        target_is_locked = (
            not cal_state["active"] and target_tracker.current_contour is not None
        )
        motion_confidence = 0.0
        motion_dx = motion_dy = 0.0
        if target_is_locked and previous_roi_gray is not None:
            motion_confidence, motion_dx, motion_dy = estimate_target_motion(
                previous_roi_gray, current_roi_gray, target_tracker.current_bbox
            )

        frames_since_segmentation = frame_index - last_segmentation_frame
        segmentation_due = (
            frame_index == 1 or frames_since_segmentation >= SEGMENTATION_INTERVAL
        )
        moved_significantly = (
            motion_confidence >= 0.55
            and (abs(motion_dx) >= TRACKING_EARLY_MOVE_PIXELS
                 or abs(motion_dy) >= TRACKING_EARLY_MOVE_PIXELS)
        )
        lost_refresh_due = (
            target_is_locked
            and target_tracker.state != "TARGET_LOST"
            and motion_confidence < 0.55
            and frames_since_segmentation >= 2
        )
        if target_is_locked and (moved_significantly or lost_refresh_due):
            segmentation_due = True
        if debug_capture_now:
            segmentation_due = True

        # Search locally while locked; resume a full Focus Zone scan after target loss.
        search_bbox = None
        if target_is_locked:
            search_bbox = target_tracker.current_bbox

        locked_skip_frame = target_is_locked and not segmentation_due
        pipeline_debug = None
        if target_is_locked and not debug_capture_now:
            clean_edges = quick_canny_preview(roi)
            top_level_contours = []
        else:
            clean_edges, top_level_contours, pipeline_debug = process_roi_outer_silhouettes(
                roi, search_bbox, return_debug=debug_capture_now and DEBUG_TARGET_DETECTION
            ) if debug_capture_now and DEBUG_TARGET_DETECTION else (
                *process_roi_outer_silhouettes(roi, search_bbox), None
            )

        ref_corners_full = None
        target_data = None
        target_contour = None
        debug_records = []
        quality_str = "LOW"
        status_type = "SEARCHING"
        is_edge_warning = False
        segmentation_ms = 0.0
        measurement_start = time.perf_counter()

        # --- Calibration Mode ---
        if cal_state["active"]:
            cal_state["frames_waited"] += 1
            card_c, card_corners, has_quad, card_status = find_reference_card(
                top_level_contours, roi_w, roi_h, with_status=True
            )

            if card_c is not None and card_corners is not None:
                cal_sample = calibrate_reference(card_corners, roi_x, roi_y, has_quad)
                if cal_sample is not None:
                    cal_state["samples"].append(cal_sample)
                    ref_corners_full = cal_sample["corners_full"]
                    cal_state["diagnostic"] = "CALIBRATION: SEARCHING CARD"
                else:
                    cal_state["diagnostic"] = "CALIBRATION: CARD REJECTED"
            else:
                cal_state["diagnostic"] = f"CALIBRATION: {card_status}"

            sample_count = len(cal_state["samples"])
            if sample_count > 0 and cal_state["diagnostic"] == "CALIBRATION: SEARCHING CARD":
                cal_state["message"] = (
                    f"CALIBRATION: SEARCHING CARD ({sample_count}/{CALIBRATION_SAMPLES_NEEDED})"
                )
            elif sample_count > 0:
                cal_state["message"] = (
                    f"{cal_state['diagnostic']} ({sample_count}/{CALIBRATION_SAMPLES_NEEDED} samples)"
                )
            else:
                cal_state["message"] = cal_state["diagnostic"]
            status_type = "TARGET_DETECTED" if sample_count > 0 else "EMPTY"

            if sample_count >= CALIBRATION_SAMPLES_NEEDED:
                # Median scale & outlier rejection
                px_w_list = [s["px_per_cm_w"] for s in cal_state["samples"]]
                px_h_list = [s["px_per_cm_h"] for s in cal_state["samples"]]
                med_px_w = float(np.median(px_w_list))
                med_px_h = float(np.median(px_h_list))

                diffs = [abs(s["px_per_cm_w"] - med_px_w) + abs(s["px_per_cm_h"] - med_px_h)
                         for s in cal_state["samples"]]
                best_sample_idx = int(np.argmin(diffs))

                calibration_data = cal_state["samples"][best_sample_idx]
                calibration_data["px_per_cm_w"] = med_px_w
                calibration_data["px_per_cm_h"] = med_px_h

                cal_state["active"] = False
                cal_state["samples"].clear()
                cal_state["message"] = "CALIBRATION: READY"
                cal_state["diagnostic"] = "CALIBRATION: READY"
                stabilizer.reset()
                target_tracker.reset()
                last_segmentation_mask = None
                last_segmentation_contour = None
                last_segmentation_frame = frame_index - SEGMENTATION_INTERVAL

                print("[OK] Calibration successfully locked from stable frames!")
                print(f"     Scale: {med_px_w:.2f} px/cm (W), {med_px_h:.2f} px/cm (H)")

            elif cal_state["frames_waited"] > CALIBRATION_TIMEOUT_FRAMES:
                cal_state["active"] = False
                cal_state["samples"].clear()
                cal_state["message"] = f"{cal_state['diagnostic']} - TIMED OUT"
                print(f"[WARN] {cal_state['message']}")

        # --- Measurement Mode ---
        else:
            if locked_skip_frame:
                if motion_confidence >= 0.55:
                    translated_contour = target_tracker.current_contour.copy()
                    translated_contour[:, 0, 0] += int(round(motion_dx))
                    translated_contour[:, 0, 1] += int(round(motion_dy))
                    bx, by, bw, bh = cv2.boundingRect(translated_contour)
                    touching_edge = (
                        bx <= 4 or by <= 4 or bx + bw >= roi_w - 4 or by + bh >= roi_h - 4
                    )
                    carried_score = max(0.16, target_tracker.current_score)
                    target_contour, is_edge_warning, status_type = target_tracker.update(
                        [(translated_contour, carried_score, touching_edge)], roi_w, roi_h
                    )
                else:
                    target_contour, is_edge_warning, status_type = target_tracker.update(
                        [], roi_w, roi_h
                    )
            else:
                segmented_contour = None
                if segmentation_due:
                    segmentation_start = time.perf_counter()
                    segmentation_bbox = search_bbox
                    if target_is_locked and (
                        abs(motion_dx) > SEGMENTATION_PADDING
                        or abs(motion_dy) > SEGMENTATION_PADDING
                        or motion_confidence < 0.55
                    ):
                        segmentation_bbox = None
                    segmentation_mask, segmented_contour = segment_center_object(
                        roi, search_bbox=segmentation_bbox
                    )
                    segmentation_ms = (time.perf_counter() - segmentation_start) * 1000.0
                    last_segmentation_frame = frame_index
                    last_segmentation_mask = segmentation_mask
                    if segmented_contour is not None:
                        last_segmentation_contour = segmented_contour.copy()

                    if DEBUG_TARGET_DETECTION and pipeline_debug is not None:
                        pipeline_debug["segmentation_mask"] = segmentation_mask

                    if target_is_locked and segmented_contour is None:
                        clean_edges, top_level_contours, pipeline_debug = process_roi_outer_silhouettes(
                            roi, search_bbox,
                            return_debug=debug_capture_now and DEBUG_TARGET_DETECTION
                        ) if debug_capture_now and DEBUG_TARGET_DETECTION else (
                            *process_roi_outer_silhouettes(roi, search_bbox), None
                        )
                    elif target_is_locked and pipeline_debug is None:
                        clean_edges = quick_canny_preview(roi)
                elif last_segmentation_contour is not None:
                    segmented_contour = last_segmentation_contour.copy()

                if segmented_contour is not None:
                    top_level_contours.append(segmented_contour)

                target_contour, ref_contour, is_edge_warning, status_type = select_primary_target(
                    top_level_contours, calibration_data, roi_w, roi_h, target_tracker, clean_edges,
                    debug_records=debug_records if DEBUG_TARGET_DETECTION else None
                )

            # Draw reference card in orange if still sitting on the table
            if ref_contour is not None:
                corners_roi, _ = get_reference_corners(ref_contour)
                ref_corners_full = corners_roi + np.array([roi_x, roi_y], dtype=np.float32)

            # Measure target object if valid and not touching boundary
            if target_contour is not None and not is_edge_warning:
                raw_target = measure_target_object(target_contour, calibration_data, roi_x, roi_y)
                if raw_target is not None:
                    if calibration_data is not None and raw_target["w_cm"] > 0:
                        stab_result = stabilizer.update(raw_target["w_cm"], raw_target["h_cm"])
                        if stab_result is not None:
                            raw_target["w_cm"] = stab_result[0]
                            raw_target["h_cm"] = stab_result[1]
                    else:
                        stab_result = stabilizer.update(raw_target["w_px"], raw_target["h_px"])
                        if stab_result is not None:
                            raw_target["w_px"] = int(round(stab_result[0]))
                            raw_target["h_px"] = int(round(stab_result[1]))

                    target_data = raw_target
                    quality_str = calculate_measurement_quality(calibration_data, target_data, stabilizer)
            else:
                stabilizer.reset()

        if target_tracker.current_contour is None and target_tracker.pending_contour is None:
            last_segmentation_mask = None
            last_segmentation_contour = None

        measurement_ms = max(
            0.0, (time.perf_counter() - measurement_start) * 1000.0 - segmentation_ms
        )

        if debug_capture_now and DEBUG_TARGET_DETECTION and pipeline_debug is not None:
            if "segmentation_mask" not in pipeline_debug and last_segmentation_mask is not None:
                pipeline_debug["segmentation_mask"] = last_segmentation_mask
            save_target_detection_debug(
                roi, pipeline_debug, top_level_contours, debug_records, target_contour
            )

        # 4. Render Fullscreen Dashboard
        render_start = time.perf_counter()
        dashboard = build_fullscreen_display(
            frame, clean_edges, roi_coords, ref_corners_full, target_data,
            calibration_data, quality_str, cal_state["message"], status_type,
            is_edge_warning, screen_w, screen_h
        )

        cv2.imshow(window_name, dashboard)
        render_ms = (time.perf_counter() - render_start) * 1000.0

        # 5. Handle Keyboard Controls
        key = cv2.waitKey(1) & 0xFF

        total_frame_ms = (time.perf_counter() - frame_start) * 1000.0
        perf_samples.append((capture_ms, segmentation_ms, measurement_ms, render_ms, total_frame_ms))
        if len(perf_samples) >= PERF_REPORT_INTERVAL:
            samples = perf_samples[-PERF_REPORT_INTERVAL:]
            average = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
            segmentation_times = [sample[1] for sample in samples if sample[1] > 0]
            segmentation_average = float(np.mean(segmentation_times)) if segmentation_times else 0.0
            fps = 1000.0 / max(average[4], 1e-6)
            print(
                f"[PERF] FPS: {fps:.1f} | Capture: {average[0]:.1f} ms | "
                f"Segmentation: {segmentation_average:.1f} ms ({len(segmentation_times)}/{len(samples)} frames) | "
                f"Measurement: {average[2]:.1f} ms | Rendering: {average[3]:.1f} ms | "
                f"Total: {average[4]:.1f} ms"
            )
            perf_samples.clear()

        previous_roi_gray = current_roi_gray

        if key == ord("q"):
            break

        elif key == ord("d") and DEBUG_TARGET_DETECTION:
            capture_debug_next_frame = True
            print("[DEBUG] Capture queued for next processed frame.")

        elif key == ord("c"):
            cal_state["active"] = True
            cal_state["samples"].clear()
            cal_state["frames_waited"] = 0
            cal_state["diagnostic"] = "CALIBRATION: SEARCHING CARD"
            cal_state["message"] = "CALIBRATION: SEARCHING CARD"
            print("[INFO] Starting calibration sampling...")

        elif key == ord("r"):
            calibration_data = None
            stabilizer.reset()
            target_tracker.reset()
            last_segmentation_mask = None
            last_segmentation_contour = None
            last_segmentation_frame = frame_index - SEGMENTATION_INTERVAL
            cal_state["active"] = False
            cal_state["samples"].clear()
            cal_state["message"] = ""
            print("[OK] Calibration reset.")

    # Cleanup cleanly
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed cleanly.")


if __name__ == "__main__":
    main()
