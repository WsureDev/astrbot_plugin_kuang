from __future__ import annotations


def connected_components(binary_mask) -> list[tuple[int, int, int, int, int]]:
    """Four-connected regions as inclusive bounds and foreground pixel count."""
    import numpy as np

    mask = np.ascontiguousarray(binary_mask, dtype=np.uint8)
    if not mask.size or not mask.any():
        return []
    try:
        import cv2
    except (ImportError, OSError):
        return _scanline_components(mask)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    return [
        (int(x), int(y), int(x + w - 1), int(y + h - 1), int(area))
        for x, y, w, h, area in stats[1:]
    ]


def _scanline_components(mask) -> list[tuple[int, int, int, int, int]]:
    """Label horizontal runs with union-find; no OpenCV dependency.

    Adjacent rows connect only where their runs overlap (four-connectivity).
    Work per run instead of visiting each foreground pixel in Python.
    """
    import numpy as np

    parents: list[int] = []
    bounds: list[list[int]] = []

    def root(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    previous = []
    for y, row in enumerate(mask):
        padded = np.pad(row != 0, (1, 1)).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded)).tolist()
        current = []
        first = 0
        for x1, end in zip(edges[::2], edges[1::2]):
            x2 = end - 1
            label = len(parents)
            parents.append(label)
            bounds.append([x1, y, x2, y, end - x1])
            while first < len(previous) and previous[first][1] < x1:
                first += 1
            index = first
            while index < len(previous) and previous[index][0] <= x2:
                other = root(previous[index][2])
                label = root(label)
                if other != label:
                    parents[other] = label
                    a, b = bounds[label], bounds[other]
                    a[:] = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                        a[4] + b[4],
                    ]
                index += 1
            current.append((x1, x2, label))
        previous = current
    return [tuple(b) for label, b in enumerate(bounds) if parents[label] == label]
