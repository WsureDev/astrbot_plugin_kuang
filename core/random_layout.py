from __future__ import annotations

import random
from typing import Any

from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)


class PerspectiveRandomBoxGenerator:
    def __init__(
        self,
        *,
        box_count: int = 5,
        random_width_ratio_min: float = 0.28,
        random_width_ratio_max: float = 0.52,
        random_height_ratio_min: float = 0.09,
        random_height_ratio_max: float = 0.55,
    ) -> None:
        self.box_count = max(1, int(box_count))
        self.random_width_ratio_min = float(random_width_ratio_min)
        self.random_width_ratio_max = float(random_width_ratio_max)
        self.random_height_ratio_min = float(random_height_ratio_min)
        self.random_height_ratio_max = float(random_height_ratio_max)

    def generate_random_layout_preview(
        self,
        image_width: int,
        image_height: int,
        *,
        box_count: int | None = None,
        seed: int | None = None,
    ) -> tuple[list[DetectionBox], dict[str, Any]]:
        target_count = self.box_count if box_count is None else max(1, int(box_count))
        rng = self._build_random_rng(image_width, image_height, target_count, seed)
        scene = self._build_perspective_scene(image_width, image_height, rng)
        boxes = self.generate_missing_boxes(
            image_width,
            image_height,
            target_count,
            [],
            rng=rng,
            scene=scene,
        )
        return boxes[:target_count], scene

    def generate_missing_boxes(
        self,
        image_width: int,
        image_height: int,
        missing_count: int,
        existing_boxes: list[DetectionBox],
        *,
        rng: random.Random | None = None,
        scene: dict[str, Any] | None = None,
    ) -> list[DetectionBox]:
        if missing_count <= 0:
            return []

        if rng is None:
            rng = self._build_random_rng(
                image_width,
                image_height,
                missing_count,
            )
        if scene is None:
            scene = self._build_perspective_scene(image_width, image_height, rng)

        generated: list[DetectionBox] = []
        max_attempts = max(80, missing_count * 80)
        empty_candidates = 0
        overlap_rejections = 0
        for _ in range(max_attempts):
            if len(generated) >= missing_count:
                break

            candidate = self._make_random_candidate(
                image_width,
                image_height,
                rng,
                scene,
            )
            if candidate is None:
                empty_candidates += 1
                continue

            all_boxes = existing_boxes + generated
            if any(self._overlaps(candidate, other) for other in all_boxes):
                overlap_rejections += 1
                continue

            generated.append(candidate)

        if len(generated) < missing_count:
            generated.extend(
                self._generate_perspective_fallback_boxes(
                    image_width,
                    image_height,
                    missing_count - len(generated),
                    existing_boxes + generated,
                    rng,
                    scene,
                )
            )
        _logger.debug(
            "[random_layout] generate_missing_boxes: target=%s produced=%s attempts=%s empty_candidates=%s overlap_rejections=%s existing=%s details=%s",
            missing_count,
            len(generated),
            max_attempts,
            empty_candidates,
            overlap_rejections,
            len(existing_boxes),
            self._summarize_boxes(generated),
        )
        return generated

    @staticmethod
    def _build_random_rng(
        image_width: int,
        image_height: int,
        box_count: int,
        seed: int | None = None,
    ) -> random.Random:
        if seed is None:
            seed = (image_width << 16) ^ image_height ^ box_count
        return random.Random(seed)

    def _build_perspective_scene(
        self,
        image_width: int,
        image_height: int,
        rng: random.Random,
    ) -> dict[str, Any]:
        horizon_y = image_height / 2.0
        hfov_degrees = 98.0
        vfov_degrees = 58.0
        focal_x = (image_width / 2.0) / self._tan_deg(hfov_degrees / 2.0)
        focal_y = (image_height / 2.0) / self._tan_deg(vfov_degrees / 2.0)
        camera_height = 1.65
        platforms: list[dict[str, Any]] = []

        platform_count = 1 + rng.randint(0, 1)
        attempts = 0
        while len(platforms) < platform_count and attempts < 40:
            attempts += 1
            radius = rng.uniform(8.0, 22.0)
            angle_rad = self._deg_to_rad(rng.uniform(-70.0, 70.0))
            x = radius * self._sin(angle_rad)
            z = radius * self._cos(angle_rad)
            width = rng.uniform(1.8, 3.2)
            depth = rng.uniform(1.2, 2.2)
            height = rng.uniform(0.45, 1.10)
            if z <= (depth / 2.0) + 1.0:
                continue
            if any(
                abs(platform["x"] - x) < ((platform["width"] + width) * 0.55)
                and abs(platform["z"] - z) < ((platform["depth"] + depth) * 0.55)
                for platform in platforms
            ):
                continue

            platforms.append(
                {
                    "x": x,
                    "z": z,
                    "width": width,
                    "depth": depth,
                    "height": height,
                }
            )

        platforms.sort(key=lambda item: (item["z"], item["x"]))
        return {
            "horizon_y": horizon_y,
            "camera_height": camera_height,
            "hfov_degrees": hfov_degrees,
            "vfov_degrees": vfov_degrees,
            "focal_x": focal_x,
            "focal_y": focal_y,
            "platforms": platforms,
        }

    @staticmethod
    def _deg_to_rad(value: float) -> float:
        return value * 0.017453292519943295

    @staticmethod
    def _sin(value: float) -> float:
        import math

        return math.sin(value)

    @staticmethod
    def _cos(value: float) -> float:
        import math

        return math.cos(value)

    @staticmethod
    def _tan_deg(value: float) -> float:
        import math

        return math.tan(math.radians(value))

    def _project_world_point(
        self,
        x: float,
        y: float,
        z: float,
        scene: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float] | None:
        if z <= 0.2:
            return None
        focal_x = float(scene["focal_x"])
        focal_y = float(scene["focal_y"])
        camera_height = float(scene["camera_height"])
        screen_x = (image_width / 2.0) + ((x * focal_x) / z)
        screen_y = (image_height / 2.0) - (((y - camera_height) * focal_y) / z)
        return screen_x, screen_y

    def _make_random_candidate(
        self,
        image_width: int,
        image_height: int,
        rng: random.Random,
        scene: dict[str, Any],
    ) -> DetectionBox | None:
        platforms = scene["platforms"]
        use_platform = bool(platforms) and rng.random() < 0.24

        if use_platform:
            platform = rng.choice(platforms)
            local_x = rng.uniform(
                -(float(platform["width"]) * 0.34),
                float(platform["width"]) * 0.34,
            )
            world_x = float(platform["x"]) + local_x
            world_z = float(platform["z"]) + rng.uniform(
                -(float(platform["depth"]) * 0.08),
                float(platform["depth"]) * 0.08,
            )
            foot_world_y = float(platform["height"])
            source = "random_platform"
        else:
            distance_bucket = rng.random()
            if distance_bucket < 0.50:
                radius = rng.uniform(3.2, 8.5)
            elif distance_bucket < 0.82:
                radius = rng.uniform(8.5, 18.0)
            else:
                radius = rng.uniform(18.0, 34.0)
            angle_rad = self._deg_to_rad(rng.uniform(-82.0, 82.0))
            world_x = radius * self._sin(angle_rad)
            world_z = radius * self._cos(angle_rad)
            foot_world_y = 0.0
            source = "random_ground"

        if world_z <= 1.0:
            return None

        foot_point = self._project_world_point(
            world_x,
            foot_world_y,
            world_z,
            scene,
            image_width,
            image_height,
        )
        if foot_point is None:
            return None
        screen_center_x, screen_foot_y = foot_point

        top_world_y = foot_world_y + rng.uniform(1.58, 1.80)
        top_point = self._project_world_point(
            world_x,
            top_world_y,
            world_z,
            scene,
            image_width,
            image_height,
        )
        if top_point is None:
            return None

        _, screen_top_y = top_point
        box_height = int(round(screen_foot_y - screen_top_y))
        if box_height < max(10, int(image_height * 0.035)):
            return None

        box_width = int(
            round(
                box_height
                * rng.uniform(
                    self.random_width_ratio_min,
                    self.random_width_ratio_max,
                )
            )
        )

        x1 = int(round(screen_center_x - (box_width / 2.0)))
        x2 = x1 + box_width
        y2 = int(round(screen_foot_y))
        y1 = y2 - box_height

        if x2 <= 0 or x1 >= image_width:
            return None
        if y2 <= 0 or y1 >= image_height:
            return None

        x1 = max(0, x1)
        x2 = min(image_width, x2)
        y1 = max(0, y1)
        y2 = min(image_height, y2)
        if x2 - x1 < 12 or y2 - y1 < 16:
            return None

        if screen_foot_y < image_height * 0.36:
            return None
        if screen_center_x < image_width * 0.01 or screen_center_x > image_width * 0.99:
            return None
        if screen_center_x < image_width * 0.14 and y1 < image_height * 0.18:
            return None
        if screen_center_x > image_width * 0.86 and y1 < image_height * 0.18:
            return None

        return DetectionBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=0.0,
            category="random",
            priority=99,
            source=source,
        )

    def _generate_perspective_fallback_boxes(
        self,
        image_width: int,
        image_height: int,
        missing_count: int,
        existing_boxes: list[DetectionBox],
        rng: random.Random,
        scene: dict[str, Any],
    ) -> list[DetectionBox]:
        fallback: list[DetectionBox] = []
        slots: list[tuple[str, Any, Any]] = []

        for platform in scene["platforms"]:
            slots.append(("platform", platform, 0.0))
            slots.append(("platform", platform, -0.26))
            slots.append(("platform", platform, 0.26))

        radius_slots = [3.5, 5.0, 7.0, 9.5, 13.0, 17.0, 22.0, 28.0, 34.0]
        angle_slots = [-84.0, -68.0, -52.0, -36.0, -20.0, -8.0, 8.0, 20.0, 36.0, 52.0, 68.0, 84.0]
        for radius in radius_slots:
            for angle in angle_slots:
                slots.append(("ground", radius, angle))

        rng.shuffle(slots)
        for slot_kind, slot_a, slot_b in slots:
            if len(fallback) >= missing_count:
                return fallback

            if slot_kind == "platform":
                platform = slot_a
                world_x = float(platform["x"]) + (float(slot_b) * float(platform["width"]) * 0.28)
                world_z = float(platform["z"])
                foot_world_y = float(platform["height"])
                source = "random_platform_fallback"
            else:
                radius = float(slot_a)
                angle_rad = self._deg_to_rad(float(slot_b))
                world_x = radius * self._sin(angle_rad)
                world_z = radius * self._cos(angle_rad)
                foot_world_y = 0.0
                source = "random_ground_fallback"

            if world_z <= 1.0:
                continue
            foot_point = self._project_world_point(
                world_x,
                foot_world_y,
                world_z,
                scene,
                image_width,
                image_height,
            )
            top_point = self._project_world_point(
                world_x,
                foot_world_y + 1.70,
                world_z,
                scene,
                image_width,
                image_height,
            )
            if foot_point is None or top_point is None:
                continue

            screen_center_x, screen_foot_y = foot_point
            _, screen_top_y = top_point
            box_height = int(round(screen_foot_y - screen_top_y))
            if box_height < max(10, int(image_height * 0.035)):
                continue

            box_width = int(
                round(
                    box_height
                    * ((self.random_width_ratio_min + self.random_width_ratio_max) / 2.0)
                )
            )

            x1 = int(round(screen_center_x - (box_width / 2.0)))
            x2 = x1 + box_width
            y2 = int(round(screen_foot_y))
            y1 = y2 - box_height
            if x2 <= 0 or x1 >= image_width or y2 <= 0 or y1 >= image_height:
                continue

            x1 = max(0, x1)
            x2 = min(image_width, x2)
            y1 = max(0, y1)
            y2 = min(image_height, y2)
            if x2 - x1 < 12 or y2 - y1 < 16:
                continue

            candidate = DetectionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=0.0,
                category="random",
                priority=99,
                source=source,
            )
            all_boxes = existing_boxes + fallback
            if any(self._overlaps(candidate, other) for other in all_boxes):
                continue
            fallback.append(candidate)
        return fallback

    @staticmethod
    def _overlaps(left: DetectionBox, right: DetectionBox) -> bool:
        return left.overlaps(right)

    @staticmethod
    def _summarize_boxes(boxes: list[DetectionBox]) -> str:
        if not boxes:
            return "<none>"
        return ", ".join(item.describe() for item in boxes)
