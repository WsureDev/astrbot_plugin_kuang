from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .arranger import DetectionArranger
from .detection_backend import DetectionBackend
from .logger import get_logger
from .models import DetectionBox, DetectionPipelineResult
from .motion_backend import MotionDetectionBackend
from .random_layout import PerspectiveRandomBoxGenerator
from .saliency_backend import SaliencyDetectionBackend
from .yolo_backend import (
    DEFAULT_ANIME_YOLO_MODEL_URL,
    DEFAULT_YOLO26N_MODEL_URL,
    AnimeYoloDetectionBackend,
    Yolo26nDetectionBackend,
)

_DETECTION_BACKENDS = {
    "yolo26n": Yolo26nDetectionBackend,
    "anime_yolo": AnimeYoloDetectionBackend,
}
_ANIMAL_CATEGORIES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
_CASCADE_YOLO26N_ANIME_BACKEND = "cascade_yolo26n_anime"
_logger = get_logger(__name__)


@dataclass(slots=True)
class _DetectorBackendPlan:
    mode: str
    primary_backend: DetectionBackend | None
    secondary_backend: DetectionBackend | None
    anime_fallback_trigger_count: int
    ignore_secondary_errors: bool


class EspBoxDetector:
    def __init__(
        self,
        *,
        box_count: int = 5,
        model_path: str,
        model_url: str = DEFAULT_YOLO26N_MODEL_URL,
        auto_download_model: bool = True,
        confidence_threshold: float = 0.25,
        random_width_ratio_min: float = 0.28,
        random_width_ratio_max: float = 0.52,
        random_height_ratio_min: float = 0.09,
        random_height_ratio_max: float = 0.55,
        nms_iou_threshold: float = 0.45,
        model_input_size: int = 640,
        enable_random_boxes: bool = True,
        backend_name: str = "yolo26n",
        enable_anime_fallback: bool = False,
        anime_model_path: str = "",
        anime_model_url: str = DEFAULT_ANIME_YOLO_MODEL_URL,
        anime_auto_download_model: bool = False,
        anime_confidence_threshold: float = 0.25,
        anime_nms_iou_threshold: float = 0.45,
        anime_model_input_size: int = 640,
        anime_fallback_trigger_count: int = 2,
        anime_merge_iou_threshold: float = 0.45,
        enable_saliency: bool = False,
        saliency_model_path: str = "",
        saliency_model_url: str = "",
        saliency_auto_download_model: bool = True,
        saliency_model_input_size: int = 320,
        saliency_threshold: float = 0.5,
        saliency_min_area_ratio: float = 0.005,
        saliency_max_boxes: int = 8,
        enable_motion: bool = False,
        motion_diff_threshold: int = 25,
        motion_min_area_ratio: float = 0.005,
        motion_max_boxes: int = 8,
        motion_downscale_long_edge: int = 240,
        motion_dilate_radius: int = 4,
        detection_backend: DetectionBackend | None = None,
    ) -> None:
        self.box_count = max(1, int(box_count))
        self.enable_random_boxes = bool(enable_random_boxes)
        self._candidate_limit = max(12, self.box_count * 4)
        _logger.debug(
            "[detector] init: backend=%s box_count=%s enable_random_boxes=%s enable_anime_fallback=%s anime_model_path=%s anime_auto_download=%s anime_trigger=%s candidate_limit=%s",
            backend_name,
            self.box_count,
            self.enable_random_boxes,
            enable_anime_fallback,
            anime_model_path or "<default>",
            anime_auto_download_model,
            anime_fallback_trigger_count,
            self._candidate_limit,
        )
        self._backend_plan = self._build_backend_plan(
            detection_backend=detection_backend,
            backend_name=backend_name,
            model_path=model_path,
            model_url=model_url,
            auto_download_model=auto_download_model,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            model_input_size=model_input_size,
            enable_anime_fallback=enable_anime_fallback,
            anime_model_path=anime_model_path,
            anime_model_url=anime_model_url,
            anime_auto_download_model=anime_auto_download_model,
            anime_confidence_threshold=anime_confidence_threshold,
            anime_nms_iou_threshold=anime_nms_iou_threshold,
            anime_model_input_size=anime_model_input_size,
            anime_fallback_trigger_count=anime_fallback_trigger_count,
        )
        self._arranger = DetectionArranger(
            box_count=self.box_count,
            anime_merge_iou_threshold=anime_merge_iou_threshold,
        )
        self._random_box_generator = PerspectiveRandomBoxGenerator(
            box_count=self.box_count,
            random_width_ratio_min=random_width_ratio_min,
            random_width_ratio_max=random_width_ratio_max,
            random_height_ratio_min=random_height_ratio_min,
            random_height_ratio_max=random_height_ratio_max,
        )

        self.enable_saliency = bool(enable_saliency)
        self._saliency_backend: SaliencyDetectionBackend | None = None
        if self.enable_saliency:
            self._saliency_backend = SaliencyDetectionBackend(
                model_path=saliency_model_path,
                model_url=saliency_model_url,
                auto_download_model=saliency_auto_download_model,
                model_input_size=saliency_model_input_size,
                binary_threshold=saliency_threshold,
                min_area_ratio=saliency_min_area_ratio,
                max_boxes=saliency_max_boxes,
            )
            _logger.info(
                "[detector] saliency stage3 enabled: model_path=%s auto_download=%s input=%s threshold=%.3f",
                saliency_model_path or "<default>",
                saliency_auto_download_model,
                saliency_model_input_size,
                saliency_threshold,
            )
        else:
            _logger.debug("[detector] saliency stage3 disabled")

        self.enable_motion = bool(enable_motion)
        self._motion_backend: MotionDetectionBackend | None = None
        if self.enable_motion:
            self._motion_backend = MotionDetectionBackend(
                diff_threshold=motion_diff_threshold,
                min_area_ratio=motion_min_area_ratio,
                max_boxes=motion_max_boxes,
                downscale_long_edge=motion_downscale_long_edge,
                dilate_radius=motion_dilate_radius,
            )
            _logger.info(
                "[detector] motion stage4 enabled: diff_threshold=%s min_area_ratio=%.4f downscale=%s dilate=%s",
                motion_diff_threshold,
                motion_min_area_ratio,
                motion_downscale_long_edge,
                motion_dilate_radius,
            )
        else:
            _logger.debug("[detector] motion stage4 disabled")

    def load_image(self, image_path: str):
        np = self._load_numpy()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))

    def analyze_from_path(self, image_path: str) -> DetectionPipelineResult:
        _logger.debug("[detector] analyze_from_path start: %s", image_path)
        image = self.load_image(image_path)
        result = self.analyze(image)
        _logger.debug(
            "[detector] analyze_from_path done: final=%s path=%s",
            len(result.final_boxes),
            image_path,
        )
        return result

    def detect_from_path(self, image_path: str) -> list[DetectionBox]:
        return self.analyze_from_path(image_path).final_boxes

    def analyze(
        self,
        image_rgb,
        *,
        previous_frame_rgb=None,
    ) -> DetectionPipelineResult:
        image_height, image_width = image_rgb.shape[:2]
        stage1_boxes = self._run_stage1(image_rgb)
        stage2_boxes = self._run_stage2(image_rgb, stage1_boxes)
        stage2_composite_boxes = self._arranger.compose_anime_boxes(stage2_boxes)
        stage3_boxes = self._run_stage3(image_rgb)
        stage4_boxes = self._run_stage4(image_rgb, previous_frame_rgb)
        arranged_boxes = self._arranger.arrange(
            stage1_boxes=stage1_boxes,
            stage2_boxes=stage2_boxes,
            stage2_composite_boxes=stage2_composite_boxes,
            stage3_boxes=stage3_boxes,
            stage4_boxes=stage4_boxes,
        )
        _logger.debug(
            "[detector] arranged detections=%s details=%s",
            len(arranged_boxes),
            self._summarize_detections(arranged_boxes),
        )

        random_boxes: list[DetectionBox] = []
        if self.enable_random_boxes and len(arranged_boxes) < self.box_count:
            missing = self.box_count - len(arranged_boxes)
            _logger.debug(
                "[detector] random box fallback triggered: arranged=%s missing=%s",
                len(arranged_boxes),
                missing,
            )
            random_boxes = [
                item.clone(
                    stage="stage5_random",
                    semantic_role="random",
                    part="random",
                )
                for item in self._random_box_generator.generate_missing_boxes(
                    image_width,
                    image_height,
                    missing,
                    arranged_boxes,
                )
            ]
            _logger.debug(
                "[detector] random boxes=%s details=%s",
                len(random_boxes),
                self._summarize_detections(random_boxes),
            )
        else:
            _logger.debug(
                "[detector] random box fallback skipped: enabled=%s arranged=%s target=%s",
                self.enable_random_boxes,
                len(arranged_boxes),
                self.box_count,
            )

        final_detections = (arranged_boxes + random_boxes)[: self.box_count]
        _logger.debug(
            "[detector] pipeline summary: stage1=%s stage2=%s stage2_composite=%s stage3=%s stage4=%s arranged=%s random=%s final=%s",
            len(stage1_boxes),
            len(stage2_boxes),
            len(stage2_composite_boxes),
            len(stage3_boxes),
            len(stage4_boxes),
            len(arranged_boxes),
            len(random_boxes),
            len(final_detections),
        )
        _logger.debug(
            "[detector] final detections=%s details=%s",
            len(final_detections),
            self._summarize_detections(final_detections),
        )
        return DetectionPipelineResult(
            stage1_boxes=stage1_boxes,
            stage2_boxes=stage2_boxes,
            stage2_composite_boxes=stage2_composite_boxes,
            stage3_boxes=stage3_boxes,
            stage4_boxes=stage4_boxes,
            arranged_boxes=arranged_boxes,
            random_boxes=random_boxes,
            final_boxes=final_detections,
        )

    def detect(self, image_rgb) -> list[DetectionBox]:
        return self.analyze(image_rgb).final_boxes

    def generate_random_layout_preview(
        self,
        image_width: int,
        image_height: int,
        *,
        box_count: int | None = None,
        seed: int | None = None,
    ) -> tuple[list[DetectionBox], dict[str, Any]]:
        return self._random_box_generator.generate_random_layout_preview(
            image_width,
            image_height,
            box_count=box_count,
            seed=seed,
        )

    def _load_numpy(self):
        for backend in (
            self._backend_plan.primary_backend,
            self._backend_plan.secondary_backend,
        ):
            if backend is None:
                continue
            load_numpy = getattr(backend, "load_numpy", None)
            if callable(load_numpy):
                return load_numpy()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少 NumPy 依赖，请先安装 requirements.txt。") from exc
        return np

    def _run_stage1(self, image_rgb) -> list[DetectionBox]:
        if self._backend_plan.mode == "anime_only":
            _logger.debug("[detector] stage1 skipped because backend mode is anime_only")
            return []
        if self._backend_plan.primary_backend is None:
            return []
        detections = self._limit_candidates(
            self._backend_plan.primary_backend.detect(image_rgb),
            stage_name="stage1_primary",
        )
        _logger.debug(
            "[detector] stage1 detections=%s details=%s",
            len(detections),
            self._summarize_detections(detections),
        )
        return detections

    def _run_stage2(
        self,
        image_rgb,
        stage1_boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        if self._backend_plan.mode == "anime_only":
            if self._backend_plan.primary_backend is None:
                return []
            detections = self._limit_candidates(
                self._backend_plan.primary_backend.detect(image_rgb),
                stage_name="stage2_anime",
            )
            _logger.debug(
                "[detector] stage2 anime-only detections=%s details=%s",
                len(detections),
                self._summarize_detections(detections),
            )
            return detections

        if self._backend_plan.mode != "yolo_with_anime_fallback":
            _logger.debug("[detector] stage2 skipped because anime fallback is disabled")
            return []

        person_count = sum(1 for item in stage1_boxes if item.category == "person")
        should_run = person_count < self._backend_plan.anime_fallback_trigger_count
        _logger.debug(
            "[detector] stage2 trigger: stage1_person_count=%s trigger_count=%s run=%s",
            person_count,
            self._backend_plan.anime_fallback_trigger_count,
            should_run,
        )
        if not should_run or self._backend_plan.secondary_backend is None:
            return []

        try:
            detections = self._limit_candidates(
                self._backend_plan.secondary_backend.detect(image_rgb),
                stage_name="stage2_anime",
            )
            _logger.debug(
                "[detector] stage2 detections=%s details=%s",
                len(detections),
                self._summarize_detections(detections),
            )
            return detections
        except Exception:
            if not self._backend_plan.ignore_secondary_errors:
                raise
            _logger.exception("[detector] stage2 anime fallback failed, skipping secondary detections")
            return []

    def _limit_candidates(
        self,
        detections: list[DetectionBox],
        *,
        stage_name: str,
    ) -> list[DetectionBox]:
        limited = list(detections)[: self._candidate_limit]
        if len(limited) < len(detections):
            _logger.debug(
                "[detector] candidate limit applied: stage=%s kept=%s dropped=%s",
                stage_name,
                len(limited),
                len(detections) - len(limited),
            )
        return [self._annotate_stage_box(item, stage_name) for item in limited]

    def _run_stage3(self, image_rgb) -> list[DetectionBox]:
        if not self.enable_saliency or self._saliency_backend is None:
            return []
        try:
            detections = self._saliency_backend.detect(image_rgb)
        except Exception:
            _logger.exception(
                "[detector] stage3 saliency detection failed, skipping"
            )
            return []
        limited = list(detections)[: self._candidate_limit]
        if len(limited) < len(detections):
            _logger.debug(
                "[detector] stage3 candidate limit applied: kept=%s dropped=%s",
                len(limited),
                len(detections) - len(limited),
            )
        _logger.debug(
            "[detector] stage3 detections=%s details=%s",
            len(limited),
            self._summarize_detections(limited),
        )
        return limited

    def _run_stage4(
        self,
        image_rgb,
        previous_frame_rgb,
    ) -> list[DetectionBox]:
        if not self.enable_motion or self._motion_backend is None:
            return []
        if previous_frame_rgb is None:
            _logger.debug(
                "[detector] stage4 motion skipped: no previous frame"
            )
            return []
        try:
            detections = self._motion_backend.detect(
                image_rgb,
                previous_frame_rgb,
            )
        except Exception:
            _logger.exception(
                "[detector] stage4 motion detection failed, skipping"
            )
            return []
        limited = list(detections)[: self._candidate_limit]
        if len(limited) < len(detections):
            _logger.debug(
                "[detector] stage4 candidate limit applied: kept=%s dropped=%s",
                len(limited),
                len(detections) - len(limited),
            )
        _logger.debug(
            "[detector] stage4 detections=%s details=%s",
            len(limited),
            self._summarize_detections(limited),
        )
        return limited

    @staticmethod
    def _annotate_stage_box(
        item: DetectionBox,
        stage_name: str,
    ) -> DetectionBox:
        part = item.part
        semantic_role = item.semantic_role
        if stage_name == "stage1_primary":
            part = part or item.category.lower()
            if not semantic_role:
                if item.category == "person":
                    semantic_role = "person_full"
                elif item.category in _ANIMAL_CATEGORIES:
                    semantic_role = "animal"
                else:
                    semantic_role = "other"
        elif stage_name == "stage2_anime":
            part = part or item.category.lower()
            if not semantic_role:
                if item.category == "Head":
                    semantic_role = "anime_head"
                elif item.category == "Torso":
                    semantic_role = "anime_torso"
                elif item.category == "Legs":
                    semantic_role = "anime_legs"
                else:
                    semantic_role = "other"
        return item.clone(
            stage=stage_name,
            part=part,
            semantic_role=semantic_role,
        )

    @staticmethod
    def _build_backend_plan(
        *,
        detection_backend: DetectionBackend | None,
        backend_name: str,
        model_path: str,
        model_url: str,
        auto_download_model: bool,
        confidence_threshold: float,
        nms_iou_threshold: float,
        model_input_size: int,
        enable_anime_fallback: bool,
        anime_model_path: str,
        anime_model_url: str,
        anime_auto_download_model: bool,
        anime_confidence_threshold: float,
        anime_nms_iou_threshold: float,
        anime_model_input_size: int,
        anime_fallback_trigger_count: int,
    ) -> _DetectorBackendPlan:
        if detection_backend is not None:
            _logger.info("[detector] using injected detection backend only")
            return _DetectorBackendPlan(
                mode="custom_backend",
                primary_backend=detection_backend,
                secondary_backend=None,
                anime_fallback_trigger_count=max(1, int(anime_fallback_trigger_count)),
                ignore_secondary_errors=True,
            )

        resolved_backend_name = str(backend_name or "yolo26n").strip().lower()
        _logger.debug(
            "[detector] build backend request: backend=%s enable_anime_fallback=%s",
            resolved_backend_name,
            enable_anime_fallback,
        )
        if enable_anime_fallback and resolved_backend_name == "yolo26n":
            resolved_backend_name = _CASCADE_YOLO26N_ANIME_BACKEND
            _logger.debug(
                "[detector] anime fallback enabled, switching backend mode to %s",
                resolved_backend_name,
            )

        if resolved_backend_name == _CASCADE_YOLO26N_ANIME_BACKEND:
            _logger.info(
                "[detector] using pipeline backend: primary=yolo26n secondary=anime_yolo trigger_count=%s",
                anime_fallback_trigger_count,
            )
            return _DetectorBackendPlan(
                mode="yolo_with_anime_fallback",
                primary_backend=Yolo26nDetectionBackend(
                    model_path=model_path,
                    model_url=model_url or DEFAULT_YOLO26N_MODEL_URL,
                    auto_download_model=auto_download_model,
                    confidence_threshold=confidence_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    model_input_size=model_input_size,
                ),
                secondary_backend=AnimeYoloDetectionBackend(
                    model_path=anime_model_path,
                    model_url=anime_model_url or DEFAULT_ANIME_YOLO_MODEL_URL,
                    auto_download_model=anime_auto_download_model,
                    confidence_threshold=anime_confidence_threshold,
                    nms_iou_threshold=anime_nms_iou_threshold,
                    model_input_size=anime_model_input_size,
                ),
                anime_fallback_trigger_count=max(1, int(anime_fallback_trigger_count)),
                ignore_secondary_errors=True,
            )

        if resolved_backend_name == "anime_yolo":
            _logger.info("[detector] using anime_yolo backend only")
            return _DetectorBackendPlan(
                mode="anime_only",
                primary_backend=AnimeYoloDetectionBackend(
                    model_path=anime_model_path,
                    model_url=anime_model_url or DEFAULT_ANIME_YOLO_MODEL_URL,
                    auto_download_model=anime_auto_download_model,
                    confidence_threshold=anime_confidence_threshold,
                    nms_iou_threshold=anime_nms_iou_threshold,
                    model_input_size=anime_model_input_size,
                ),
                secondary_backend=None,
                anime_fallback_trigger_count=max(1, int(anime_fallback_trigger_count)),
                ignore_secondary_errors=True,
            )

        backend_cls = _DETECTION_BACKENDS.get(resolved_backend_name)
        if backend_cls is None:
            supported = ", ".join(
                sorted((*_DETECTION_BACKENDS.keys(), _CASCADE_YOLO26N_ANIME_BACKEND))
            )
            raise RuntimeError(
                f"不支持的识别器后端: {resolved_backend_name}。当前支持: {supported}"
            )
        _logger.info("[detector] using single backend: %s", resolved_backend_name)
        return _DetectorBackendPlan(
            mode="single_backend",
            primary_backend=backend_cls(
                model_path=model_path,
                model_url=model_url or DEFAULT_YOLO26N_MODEL_URL,
                auto_download_model=auto_download_model,
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold,
                model_input_size=model_input_size,
            ),
            secondary_backend=None,
            anime_fallback_trigger_count=max(1, int(anime_fallback_trigger_count)),
            ignore_secondary_errors=True,
        )

    @staticmethod
    def _summarize_detections(detections: list[DetectionBox]) -> str:
        if not detections:
            return "<none>"
        return ", ".join(item.describe() for item in detections)
