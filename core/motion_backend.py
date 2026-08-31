from __future__ import annotations

from collections import deque
from typing import Any

from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)


class MotionDetectionBackend:
    """基于帧差的运动检测后端，专门用于 GIF 的移动物体标注。

    没有模型，纯 numpy 运算：
      1. 前后两帧转灰度、求绝对差；
      2. 差值 >= diff_threshold 的像素判定为运动像素；
      3. 对二值 mask 做一次膨胀，把角色轮廓 + 内部零散色块粘成整块；
      4. 4-连通域取每块的外接框；
      5. 太小的运动块（避免闪烁噪点）被过滤。

    融合规则统一在 arranger 里处理，本 backend 只负责产 `semantic_role="motion"` 的候选框。
    """

    source_name = "motion"

    def __init__(
        self,
        *,
        diff_threshold: int = 25,
        min_area_ratio: float = 0.005,
        max_boxes: int = 8,
        downscale_long_edge: int = 240,
        edge_padding_ratio: float = 0.01,
        dilate_radius: int = 4,
    ) -> None:
        self.diff_threshold = max(1, int(diff_threshold))
        self.min_area_ratio = max(0.0, float(min_area_ratio))
        self.max_boxes = max(1, int(max_boxes))
        self.downscale_long_edge = max(64, int(downscale_long_edge))
        self.edge_padding_ratio = max(0.0, float(edge_padding_ratio))
        self.dilate_radius = max(0, int(dilate_radius))
        self._np: Any | None = None
        _logger.debug(
            "[%s] backend init: diff_threshold=%s min_area_ratio=%.4f max_boxes=%s"
            " downscale_long_edge=%s dilate_radius=%s",
            self.source_name,
            self.diff_threshold,
            self.min_area_ratio,
            self.max_boxes,
            self.downscale_long_edge,
            self.dilate_radius,
        )

    def detect(
        self,
        current_frame_rgb,
        previous_frame_rgb,
    ) -> list[DetectionBox]:
        if previous_frame_rgb is None:
            return []
        np = self.load_numpy()
        current = np.asarray(current_frame_rgb)
        previous = np.asarray(previous_frame_rgb)
        if current.shape != previous.shape:
            _logger.debug(
                "[%s] shape mismatch, skipping motion detection: current=%s previous=%s",
                self.source_name,
                current.shape,
                previous.shape,
            )
            return []
        image_height, image_width = current.shape[:2]

        # 下采样加速；帧差阈值本来就是像素级噪声容忍度，尺寸缩小影响不大
        scale = self._compute_downscale(image_width, image_height)
        if scale < 1.0:
            work_width = max(1, int(round(image_width * scale)))
            work_height = max(1, int(round(image_height * scale)))
            current_small = self._resize_rgb(current, work_width, work_height)
            previous_small = self._resize_rgb(previous, work_width, work_height)
        else:
            work_width = image_width
            work_height = image_height
            current_small = current
            previous_small = previous

        current_gray = self._to_gray(current_small)
        previous_gray = self._to_gray(previous_small)
        diff = np.abs(current_gray.astype(np.int16) - previous_gray.astype(np.int16))
        motion_mask = diff >= self.diff_threshold
        raw_motion_pixels = int(motion_mask.sum())
        motion_mask = self._dilate(motion_mask, self.dilate_radius)
        _logger.debug(
            "[%s] frame diff: image=%sx%s work=%sx%s raw_motion_pixels=%s"
            " dilated_motion_pixels=%s mean_diff=%.2f dilate_radius=%s",
            self.source_name,
            image_width,
            image_height,
            work_width,
            work_height,
            raw_motion_pixels,
            int(motion_mask.sum()),
            float(diff.mean()),
            self.dilate_radius,
        )
        components = self._extract_connected_components(motion_mask)
        min_area_px = max(1, int(work_width * work_height * self.min_area_ratio))
        scale_x = image_width / max(1, work_width)
        scale_y = image_height / max(1, work_height)
        pad_x = int(round(image_width * self.edge_padding_ratio))
        pad_y = int(round(image_height * self.edge_padding_ratio))

        boxes: list[DetectionBox] = []
        for min_x, min_y, max_x, max_y, area in components:
            if area < min_area_px:
                continue
            x1 = int(round(min_x * scale_x)) - pad_x
            y1 = int(round(min_y * scale_y)) - pad_y
            x2 = int(round((max_x + 1) * scale_x)) + pad_x
            y2 = int(round((max_y + 1) * scale_y)) + pad_y
            x1 = max(0, min(image_width, x1))
            y1 = max(0, min(image_height, y1))
            x2 = max(0, min(image_width, x2))
            y2 = max(0, min(image_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(area / max(1, work_width * work_height))
            boxes.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    category="motion",
                    priority=0,
                    source=self.source_name,
                    stage="stage4_motion",
                    part="motion",
                    semantic_role="motion",
                )
            )

        boxes.sort(key=lambda item: -item.area)
        limited = boxes[: self.max_boxes]
        _logger.debug(
            "[%s] detect: raw_components=%s kept=%s dropped=%s",
            self.source_name,
            len(components),
            len(limited),
            len(boxes) - len(limited),
        )
        return limited

    def load_numpy(self):
        if self._np is None:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("缺少 NumPy 依赖，请先安装 requirements.txt。") from exc
            self._np = np
        return self._np

    # -- helpers ---------------------------------------------------------

    def _compute_downscale(self, width: int, height: int) -> float:
        long_edge = max(width, height)
        if long_edge <= self.downscale_long_edge:
            return 1.0
        return self.downscale_long_edge / float(long_edge)

    def _resize_rgb(self, image, target_width: int, target_height: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc
        np = self.load_numpy()
        resized = Image.fromarray(image).resize(
            (target_width, target_height),
            Image.BILINEAR,
        )
        return np.asarray(resized)

    def _to_gray(self, image_rgb):
        np = self.load_numpy()
        # 常用亮度加权
        weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        return (image_rgb.astype(np.float32) * weights).sum(axis=-1).astype(np.uint8)

    def _dilate(self, binary_mask, radius: int):
        """对二值 mask 做半径为 radius 的方形核膨胀。

        用 numpy shift-OR + 可分离核实现，避免拉入 scipy/cv2：
          - 水平膨胀 radius 次，每次做一次 3 像素窗口 OR；
          - 再做垂直方向的 radius 次 3 像素窗口 OR。
        等价于一次 (2*radius+1) 方形核膨胀。radius=0 时直接返回原图。
        """
        if radius <= 0:
            return binary_mask
        np = self.load_numpy()
        result = np.asarray(binary_mask, dtype=bool)

        # 水平膨胀
        for _ in range(radius):
            shifted = result.copy()
            shifted[:, 1:] |= result[:, :-1]
            shifted[:, :-1] |= result[:, 1:]
            result = shifted

        # 垂直膨胀
        for _ in range(radius):
            shifted = result.copy()
            shifted[1:, :] |= result[:-1, :]
            shifted[:-1, :] |= result[1:, :]
            result = shifted

        return result

    def _extract_connected_components(
        self,
        binary_mask,
    ) -> list[tuple[int, int, int, int, int]]:
        np = self.load_numpy()
        mask = np.ascontiguousarray(binary_mask, dtype=bool).copy()
        height, width = mask.shape
        fg_count = int(mask.sum())
        if fg_count == 0:
            return []
        if fg_count / max(1, height * width) > 0.92:
            return [(0, 0, width - 1, height - 1, fg_count)]

        components: list[tuple[int, int, int, int, int]] = []
        rows = [row.tolist() for row in mask]
        for seed_y in range(height):
            row = rows[seed_y]
            for seed_x in range(width):
                if not row[seed_x]:
                    continue
                queue: deque[tuple[int, int]] = deque()
                queue.append((seed_y, seed_x))
                rows[seed_y][seed_x] = False
                min_x = max_x = seed_x
                min_y = max_y = seed_y
                area = 0
                while queue:
                    y, x = queue.popleft()
                    area += 1
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    if y + 1 < height and rows[y + 1][x]:
                        rows[y + 1][x] = False
                        queue.append((y + 1, x))
                    if y - 1 >= 0 and rows[y - 1][x]:
                        rows[y - 1][x] = False
                        queue.append((y - 1, x))
                    if x + 1 < width and rows[y][x + 1]:
                        rows[y][x + 1] = False
                        queue.append((y, x + 1))
                    if x - 1 >= 0 and rows[y][x - 1]:
                        rows[y][x - 1] = False
                        queue.append((y, x - 1))
                components.append((min_x, min_y, max_x, max_y, area))
        return components
