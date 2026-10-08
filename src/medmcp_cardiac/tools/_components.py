"""Connected components on 2D binary masks, in plain numpy.

Two tools need it -- keeping the long-axis model's largest left ventricle, and
counting calcified lesions slice by slice -- and neither needs more than an
8-connected labelling of a small mask, so this stays dependency-free rather than
pulling in scipy for one function.
"""

from collections import deque
from typing import Any

import numpy as np

_NEIGHBOURS_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def label_components_2d(mask: Any) -> tuple[Any, int]:
    """Label the 8-connected components of a 2D boolean mask.

    Args:
        mask: a 2D array; non-zero is foreground.

    Returns:
        ``(labels, n)``: an int32 array with ``0`` for background and ``1..n`` per
        component, and the component count.
    """
    fg: Any = np.asarray(mask).astype(bool)
    if fg.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {fg.shape}")
    labels: Any = np.zeros(fg.shape, dtype=np.int32)
    height, width = fg.shape
    n = 0
    for start in zip(*np.nonzero(fg), strict=True):
        if labels[start]:
            continue
        n += 1
        labels[start] = n
        queue: deque[tuple[int, int]] = deque([(int(start[0]), int(start[1]))])
        while queue:
            y, x = queue.popleft()
            for dy, dx in _NEIGHBOURS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and fg[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    queue.append((ny, nx))
    return labels, n


def largest_component_2d(mask: Any) -> Any:
    """The largest 8-connected component of a 2D mask (all-False if the mask is empty)."""
    labels, n = label_components_2d(mask)
    if n == 0:
        return np.zeros(np.asarray(mask).shape, dtype=bool)
    counts: Any = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def components_touching_2d(mask: Any, anchor: Any) -> Any:
    """The components of ``mask`` that touch ``anchor`` (8-connectivity), as one mask."""
    labels, n = label_components_2d(mask)
    if n == 0:
        return np.zeros(np.asarray(mask).shape, dtype=bool)
    anchor_arr: Any = np.asarray(anchor).astype(bool)
    grown: Any = anchor_arr.copy()
    for dy, dx in _NEIGHBOURS_8:
        grown |= np.roll(np.roll(anchor_arr, dy, axis=0), dx, axis=1)
    touching = {int(v) for v in np.unique(labels[grown]) if v}
    return np.isin(labels, list(touching)) if touching else np.zeros(labels.shape, dtype=bool)
