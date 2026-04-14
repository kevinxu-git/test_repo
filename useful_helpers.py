import torch

def mask_bboxes(image, boxes, mode="white"):
    """
    image: tensor (3, H, W)
    boxes: tensor (N, 4) en XYXY (pixels)
    mode: "white" ou "interpolation"
    """
    img = image.clone()
    _, H, W = img.shape

    for box in boxes:
        x1, y1, x2, y2 = box.int()

        # clamp safe
        x1 = max(1, x1)
        y1 = max(1, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        if mode == "white":
            img[:, y1:y2, x1:x2] = 1.0

        elif mode == "interpolation":
            img = _interpolate_region(img, x1, y1, x2, y2)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return img


def _interpolate_region(img, x1, y1, x2, y2):
    """
    Remplit la bbox par interpolation à partir des bords
    """
    region_h = y2 - y1
    region_w = x2 - x1

    # bords
    top = img[:, y1 - 1, x1:x2]      # (3, W)
    bottom = img[:, y2, x1:x2]
    left = img[:, y1:y2, x1 - 1]     # (3, H)
    right = img[:, y1:y2, x2]

    for i in range(region_h):
        alpha = i / max(region_h - 1, 1)

        # interpolation verticale
        row_tb = (1 - alpha) * top + alpha * bottom  # (3, W)

        # interpolation horizontale
        left_val = left[:, i].unsqueeze(1)   # (3,1)
        right_val = right[:, i].unsqueeze(1)

        t = torch.linspace(0, 1, region_w, device=img.device).unsqueeze(0)
        row_lr = (1 - t) * left_val + t * right_val

        # mélange des deux interpolations
        row = 0.5 * row_tb + 0.5 * row_lr

        img[:, y1 + i, x1:x2] = row

    return img




