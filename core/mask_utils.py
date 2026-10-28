from __future__ import annotations


def connected_components(binary_mask) -> list[tuple[int, int, int, int, int]]:
    """Four-connected regions as inclusive bounds and foreground pixel count."""
    import cv2
    import numpy as np

    mask = np.ascontiguousarray(binary_mask, dtype=np.uint8)
    if not mask.size or not mask.any():
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    return [
        (int(x), int(y), int(x + w - 1), int(y + h - 1), int(area))
        for x, y, w, h, area in stats[1:]
    ]
