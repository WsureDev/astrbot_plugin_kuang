import unittest

from core.arranger import DetectionArranger
from core.models import DetectionBox


def make_box(
    coords,
    *,
    category="object",
    role="",
    stage="",
    score=0.5,
):
    return DetectionBox(
        *coords,
        score,
        category,
        0,
        "test",
        stage=stage,
        semantic_role=role,
    )


class DetectionArrangerTests(unittest.TestCase):
    def test_person_has_priority_over_overlapping_saliency(self):
        arranger = DetectionArranger(box_count=5)
        selected = arranger.arrange(
            stage1_boxes=[
                make_box(
                    (10, 10, 100, 100),
                    category="person",
                    role="person_full",
                    stage="stage1_primary",
                )
            ],
            stage2_boxes=[],
            stage2_composite_boxes=[],
            stage3_boxes=[
                make_box(
                    (12, 12, 98, 98),
                    category="salient",
                    role="salient",
                    stage="stage3_salient",
                )
            ],
            stage4_boxes=[],
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].semantic_role, "person_full")

    def test_motion_and_saliency_can_coexist_when_disjoint(self):
        arranger = DetectionArranger(box_count=5)
        selected = arranger.arrange(
            stage1_boxes=[],
            stage2_boxes=[],
            stage2_composite_boxes=[],
            stage3_boxes=[
                make_box(
                    (0, 0, 30, 30),
                    category="salient",
                    role="salient",
                    stage="stage3_salient",
                )
            ],
            stage4_boxes=[
                make_box(
                    (60, 60, 100, 100),
                    category="motion",
                    role="motion",
                    stage="stage4_motion",
                )
            ],
        )

        self.assertEqual({box.semantic_role for box in selected}, {"motion", "salient"})

    def test_anime_merge_threshold_does_not_control_generic_stage_conflicts(self):
        arranger = DetectionArranger(
            box_count=5,
            overlap_iou_threshold=0.8,
            anime_merge_iou_threshold=0.3,
        )
        stage1_box = make_box(
            (0, 0, 100, 100),
            category="car",
            stage="stage1_primary",
        )
        overlapping_box = make_box((30, 0, 130, 100))

        anime_selected = arranger.arrange(
            stage1_boxes=[stage1_box],
            stage2_boxes=[
                overlapping_box.clone(
                    category="Head",
                    stage="stage2_anime",
                    semantic_role="anime_head",
                )
            ],
            stage2_composite_boxes=[],
        )
        generic_selected = arranger.arrange(
            stage1_boxes=[stage1_box],
            stage2_boxes=[],
            stage2_composite_boxes=[],
            stage3_boxes=[
                overlapping_box.clone(
                    category="salient",
                    stage="stage3_salient",
                    semantic_role="salient",
                )
            ],
        )

        self.assertEqual(len(anime_selected), 1)
        self.assertEqual(len(generic_selected), 2)


if __name__ == "__main__":
    unittest.main()
