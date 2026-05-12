import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LaneDetector:
    """
    Detects lane lines and produces a polygonal road mask.
    Pure classical CV — no DL model required.
    """

    def __init__(self, config):
        self.config = config

    #Public API─────────

    def detect_lanes(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Detect lane line segments using Canny edges + HoughLinesP.

        Steps:
          1. CLAHE contrast enhancement (night / tunnel / overcast robustness)
          2. Grayscale + Gaussian blur to suppress texture noise
          3. Adaptive Canny (thresholds computed from image median — Otsu-like)
          4. Mask to trapezoidal ROI (lower road area only)
          5. Probabilistic Hough Transform to extract segments

        Adaptive Canny:
          Fixed thresholds (50/150) break at night (too few edges) and in bright sun
          (too many edges from road texture). Computing thresholds from the image
          median keeps them calibrated to the actual scene brightness.
        """
        h, w = frame_bgr.shape[:2]

        #Step 1: CLAHE on lightness channel for night / low-contrast scenes
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        #Step 2: Grayscale + Gaussian blur
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        #Step 3: Adaptive Canny — thresholds scale with scene brightness
        median_val = float(np.median(blurred))
        sigma = 0.33
        lower = max(0,   int((1.0 - sigma) * median_val))
        upper = min(255, int((1.0 + sigma) * median_val))
        #Clamp to sane range so very dark/light scenes don't go degenerate
        lower = max(20, lower)
        upper = max(lower + 30, upper)
        edges = cv2.Canny(blurred, lower, upper)

        #Step 4: Keep only the trapezoidal road region
        roi_mask = self._roi_trapezoid(h, w)
        roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

        #Step 5: Hough line detection
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,          #lowered from 40: catches fainter worn markings
            minLineLength=25,      #lowered from 30: catches shorter dashed lines
            maxLineGap=150,        #increased from 120: bridges wider dashed gaps
        )

        if lines is None:
            return []

        #Flatten from [[x1,y1,x2,y2]] → [x1,y1,x2,y2]
        return [line[0] for line in lines]

    def create_road_mask(
        self,
        frame_bgr: np.ndarray,
        lanes: List[np.ndarray],
    ) -> np.ndarray:
        """
        Build a binary road mask from the detected lane segments.

        If clear left+right lane boundaries are found, the mask is the
        filled polygon between them.  Otherwise falls back to a conservative
        static trapezoid covering the expected road area.

        Args:
            frame_bgr: Input BGR frame (for dimensions)
            lanes:     Lane segments from detect_lanes()

        Returns:
            Binary mask, same H×W as frame_bgr, values in {0, 255}
        """
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not lanes:
            #No lanes detected — fall back to conservative trapezoid
            return self._fallback_trapezoid_mask(h, w)

        #Classify segments as left / right lane boundaries
        left_pts, right_pts = self._separate_lanes(lanes, w, h)

        if left_pts and right_pts:
            #Fit a line through each side and build road polygon
            left_line = self._fit_lane_line(left_pts, h)
            right_line = self._fit_lane_line(right_pts, h)

            if left_line is not None and right_line is not None:
                poly = self._build_road_polygon(left_line, right_line, h)
                cv2.fillPoly(mask, [poly], 255)
                return mask

        #Partial detection — at least one side missing
        logger.debug(
            "Lane mask fallback: left_pts=%d right_pts=%d",
            len(left_pts), len(right_pts),
        )
        return self._fallback_trapezoid_mask(h, w)

    def apply_mask(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Zero out all pixels outside the road mask.

        Passing the masked frame to YOLO prevents it from detecting
        potholes in irrelevant regions (sky, buildings, sidewalks).

        Args:
            frame_bgr: Input BGR frame
            mask:      Binary road mask (0/255)

        Returns:
            Frame with non-road pixels set to black (0, 0, 0)
        """
        return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    def draw_lane_overlay(
        self,
        frame_bgr: np.ndarray,
        lanes: List[np.ndarray],
        road_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotate a frame with fitted lane lines and an optional road mask
        overlay (semi-transparent green tint).

        Args:
            frame_bgr:  Input BGR frame
            lanes:      Raw lane segments from detect_lanes()
            road_mask:  Binary road mask to draw as transparent fill

        Returns:
            Annotated copy of frame_bgr
        """
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        #Semi-transparent green road-mask overlay
        if road_mask is not None and np.any(road_mask):
            overlay = annotated.copy()
            green_tint = np.zeros_like(annotated)
            green_tint[:, :, 1] = 180      #green channel only
            #Blend: 70% original + 30% green tint inside mask
            blended = cv2.addWeighted(overlay, 0.70, green_tint, 0.30, 0)
            annotated[road_mask == 255] = blended[road_mask == 255]

        if lanes:
            left_pts, right_pts = self._separate_lanes(lanes, w, h)

            #Draw fitted left lane line (yellow)
            if left_pts:
                left_line = self._fit_lane_line(left_pts, h)
                if left_line is not None:
                    x1, y1, x2, y2 = left_line
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

            #Draw fitted right lane line (yellow)
            if right_pts:
                right_line = self._fit_lane_line(right_pts, h)
                if right_line is not None:
                    x1, y1, x2, y2 = right_line
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        return annotated

    #Private: ROI helpers───

    @staticmethod
    def _roi_trapezoid(h: int, w: int) -> np.ndarray:
        """
        Create a trapezoidal ROI that focuses on the road ahead.

        Covers roughly the lower 45% of the image, widening toward
        the bottom to match a forward-facing camera perspective.
        Tuned for typical dashcam / front-facing footage.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        #Vertices: [bottom-left, top-left, top-right, bottom-right]
        roi_pts = np.array([[
            (int(w * 0.05), h),              #bottom-left corner
            (int(w * 0.40), int(h * 0.55)),  #top-left (toward vanishing point)
            (int(w * 0.60), int(h * 0.55)),  #top-right
            (int(w * 0.95), h),              #bottom-right corner
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        return mask

    @staticmethod
    def _fallback_trapezoid_mask(h: int, w: int) -> np.ndarray:
        """
        Conservative static road-area mask used when lane detection fails.
        Covers the typical road region for a forward-facing camera.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.35), int(h * 0.55)),
            (int(w * 0.65), int(h * 0.55)),
            (int(w * 0.95), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    #Private: Lane classification & fitting

    @staticmethod
    def _separate_lanes(
        lanes: List[np.ndarray],
        w: int,
        h: int,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Classify raw Hough segments as left-lane or right-lane points.

        Classification rules (image coordinate system, y increases downward):
          - Left lane:  negative slope AND midpoint left of image centre
          - Right lane: positive slope AND midpoint right of image centre
          - Near-horizontal lines (|slope| < 0.3) are discarded (shadows, cracks)

        Returns:
            left_pts:  (x, y) point list from left-lane segments
            right_pts: (x, y) point list from right-lane segments
        """
        cx = w // 2
        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        for x1, y1, x2, y2 in lanes:
            dx = x2 - x1
            if dx == 0:
                continue  #vertical; skip
            slope = (y2 - y1) / dx

            #Discard near-horizontal segments (not lane markers)
            if abs(slope) < 0.3:
                continue

            mid_x = (x1 + x2) / 2.0

            if slope < 0 and mid_x < cx:
                #Negative slope + left side → left lane boundary
                left_pts.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and mid_x > cx:
                #Positive slope + right side → right lane boundary
                right_pts.extend([(x1, y1), (x2, y2)])

        return left_pts, right_pts

    @staticmethod
    def _fit_lane_line(
        pts: List[Tuple[int, int]],
        h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Fit a single line through a cloud of (x, y) lane points using polyfit,
        then extrapolate it from the bottom of the frame to the horizon.

        We fit x = f(y) (i.e. x as a function of y) because lane lines are
        closer to vertical than horizontal, making the fit numerically stable.

        Args:
            pts: List of (x, y) points belonging to one lane side
            h:   Frame height (pixels)

        Returns:
            (x_bottom, y_bottom, x_top, y_top) extrapolated endpoints,
            or None if the fit fails.
        """
        if len(pts) < 2:
            return None

        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)

        try:
            #Degree-1 polynomial: x = m*y + b
            m, b = np.polyfit(ys, xs, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        y_bottom = h
        y_top = int(h * 0.55)   #horizon line at ~55% from top

        x_bottom = int(m * y_bottom + b)
        x_top = int(m * y_top + b)

        return x_bottom, y_bottom, x_top, y_top

    @staticmethod
    def _build_road_polygon(
        left_line: Tuple[int, int, int, int],
        right_line: Tuple[int, int, int, int],
        h: int,
    ) -> np.ndarray:
        """
        Construct the filled road polygon from left and right lane endpoints.

        Polygon vertices (counter-clockwise):
          bottom-left → top-left → top-right → bottom-right

        Args:
            left_line:  (x_bot, y_bot, x_top, y_top) for left lane
            right_line: (x_bot, y_bot, x_top, y_top) for right lane

        Returns:
            (4, 2) int32 array suitable for cv2.fillPoly
        """
        lx_bot, ly_bot, lx_top, ly_top = left_line
        rx_bot, ry_bot, rx_top, ry_top = right_line

        return np.array([
            [lx_bot, ly_bot],   #bottom-left  (left lane, bottom)
            [lx_top, ly_top],   #top-left     (left lane, horizon)
            [rx_top, ry_top],   #top-right    (right lane, horizon)
            [rx_bot, ry_bot],   #bottom-right (right lane, bottom)
        ], dtype=np.int32)
