import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

#Cityscapes class IDs
_CS_ROAD     = 0
_CS_SIDEWALK = 1

_YOLOSEG_ROAD_CLASSES = {0}   #class 0 = road in Cityscapes-trained seg models


class RoadSegmenter:
    """
    Road surface segmenter — returns uint8 binary mask (0/255).

    Includes optical flow warping for temporal propagation between
    full-inference frames (reduces per-frame compute by ~2/3).
    """

    def __init__(self, model_manager, config):
        self.mm     = model_manager
        self.config = config
        self._seg   = getattr(model_manager, "segmentation", None)

        #Optical flow state for mask warping
        self._prev_gray:  Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None

    #Public API────────────

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run segmentation and update optical flow state."""
        self._prev_frame = frame_bgr.copy()
        self._prev_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return self._run_backend(frame_bgr)

    def warp_mask(
        self,
        cached_mask: np.ndarray,
        current_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Propagate a stale segmentation mask to the current frame using
        sparse Lucas-Kanade optical flow.

        At 1-2 frames stale this is near-perfect.
        At 3+ frames degradation is graceful — acceptable for road masks
        because road surface moves slowly relative to ego-vehicle forward motion.
        """
        if cached_mask is None or self._prev_gray is None:
            return cached_mask

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        #Detect corners in previous frame
        corners = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=60, qualityLevel=0.01, minDistance=10
        )
        if corners is None or len(corners) < 4:
            self._prev_gray = curr_gray
            return cached_mask

        #Lucas-Kanade optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray, corners, None,
            winSize=(15, 15), maxLevel=2,
        )
        if next_pts is None:
            self._prev_gray = curr_gray
            return cached_mask

        good_prev = corners[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]

        if len(good_prev) < 4:
            self._prev_gray = curr_gray
            return cached_mask

        #Homography from previous to current frame
        M, _ = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 3.0)
        self._prev_gray = curr_gray

        if M is None:
            return cached_mask

        h, w = cached_mask.shape[:2]
        warped = cv2.warpPerspective(
            cached_mask, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped

    #Backend dispatch──────

    def _run_backend(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._seg is None:
            return self._geometric_mask(frame_bgr)

        backend = self._seg[0] if isinstance(self._seg, (tuple, list)) else ""

        if backend == "yolov8seg":
            return self._yolov8seg_mask(frame_bgr)
        elif backend == "segformer":
            return self._segformer_mask(frame_bgr)
        elif backend == "deeplabv3":
            return self._deeplabv3_mask(frame_bgr)
        else:
            return self._geometric_mask(frame_bgr)

    #YOLOv8-seg────────────

    def _yolov8seg_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        YOLOv8l-seg with Cityscapes weights.
        Combines all road-class instance masks into a single binary road mask.
        Much faster than SegFormer (~22ms vs ~180ms) at similar quality.
        """
        _, model = self._seg
        h, w = frame_bgr.shape[:2]
        road_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            results = model(
                frame_bgr,
                imgsz   = min(w, 640),
                device  = next(model.parameters()).device,
                verbose = False,
                half    = True,
            )
            for r in results:
                if r.masks is None:
                    continue
                for i, (cls_id, mask_data) in enumerate(
                    zip(r.boxes.cls.int().tolist(), r.masks.data)
                ):
                    if cls_id in _YOLOSEG_ROAD_CLASSES:
                        m = mask_data.cpu().numpy()
                        m_resized = cv2.resize(
                            (m > 0.5).astype(np.uint8) * 255,
                            (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        road_mask = cv2.bitwise_or(road_mask, m_resized)
        except Exception as e:
            logger.warning("YOLOv8-seg inference error: %s — using geometric fallback", e)
            return self._geometric_mask(frame_bgr)

        if not np.any(road_mask):
            return self._geometric_mask(frame_bgr)

        return self._morpho_clean(road_mask)

    #SegFormer─────────────

    def _segformer_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        _, model, extractor = self._seg
        h, w = frame_bgr.shape[:2]

        #CLAHE contrast enhancement for night/shadow robustness
        lab  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        inputs = extractor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[pred == _CS_ROAD] = 255

        return self._morpho_clean(road_mask)

    #DeepLabV3────────────

    def _deeplabv3_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        _, model = self._seg
        h, w = frame_bgr.shape[:2]

        infer_w = min(w, 512)
        scale   = infer_w / w
        infer_h = int(h * scale)
        resized = cv2.resize(frame_bgr, (infer_w, infer_h))

        rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).unsqueeze(0)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        tensor = tensor.to(next(model.parameters()).device)

        with torch.inference_mode():
            out  = model(tensor)["out"]
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        bg_mask  = (pred == 0).astype(np.uint8) * 255
        bg_full  = cv2.resize(bg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_and(bg_full, self._color_road_mask(frame_bgr))

        return self._morpho_clean(combined)

    #Geometric fallback────

    def _geometric_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        color_mask = self._color_road_mask(frame_bgr)
        roi_mask   = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h * 0.40):, :] = 255
        road_mask = cv2.bitwise_and(color_mask, roi_mask)
        return self._morpho_clean(road_mask)

    #Helpers 

    @staticmethod
    def _color_road_mask(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 60, 180]))
        tan_mask  = cv2.inRange(hsv, np.array([10, 10, 50]), np.array([30, 80, 160]))
        return cv2.bitwise_or(gray_mask, tan_mask)

    @staticmethod
    def _morpho_clean(mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    @staticmethod
    def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        overlay = frame_bgr.copy()
        green = np.zeros_like(frame_bgr)
        green[:, :, 1] = 200
        overlay[mask == 255] = cv2.addWeighted(
            frame_bgr, 1 - alpha, green, alpha, 0
        )[mask == 255]
        return overlay
