import numpy as np

def update_bbox(bbox: list, action: int, step_size: float = 0.05, zoom_factor: float = 0.1) -> list:
    """
    Update normalized bounding box [xmin, ymin, xmax, ymax] based on 7 discrete actions.
    """
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin

    if action == 0: # MOVE_LEFT
        xmin -= step_size * w
        xmax -= step_size * w
    elif action == 1: # MOVE_RIGHT
        xmin += step_size * w
        xmax += step_size * w
    elif action == 2: # MOVE_UP
        ymin -= step_size * h
        ymax -= step_size * h
    elif action == 3: # MOVE_DOWN
        ymin += step_size * h
        ymax += step_size * h
    elif action == 4: # ZOOM_IN
        dw = w * zoom_factor / 2.0
        dh = h * zoom_factor / 2.0
        xmin += dw
        xmax -= dw
        ymin += dh
        ymax -= dh
    elif action == 5: # ZOOM_OUT
        dw = w * zoom_factor / 2.0
        dh = h * zoom_factor / 2.0
        xmin -= dw
        xmax += dw
        ymin -= dh
        ymax += dh

    # Ensure box remains valid & within [0, 1]
    min_size = 0.05
    if xmax - xmin < min_size:
        cx = (xmin + xmax) / 2.0
        xmin = cx - min_size / 2.0
        xmax = cx + min_size / 2.0
    if ymax - ymin < min_size:
        cy = (ymin + ymax) / 2.0
        ymin = cy - min_size / 2.0
        ymax = cy + min_size / 2.0

    xmin = max(0.0, min(1.0 - min_size, xmin))
    xmax = min(1.0, max(min_size, xmax))
    ymin = max(0.0, min(1.0 - min_size, ymin))
    ymax = min(1.0, max(min_size, ymax))

    return [xmin, ymin, xmax, ymax]
