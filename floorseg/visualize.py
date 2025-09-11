import cv2
import numpy as np

def apply_colour_overlay(img_bgr: np.ndarray, mask: np.ndarray, colour_bgr, strength=0.7) -> np.ndarray:
    """
    Apply a solid colour overlay to the masked floor area while preserving brightness.
    """
    overlay = np.full_like(img_bgr, colour_bgr, dtype=np.uint8)

    # Blend with original using strength
    blended = cv2.addWeighted(overlay, strength, img_bgr, 1 - strength, 0)

    result = img_bgr.copy()
    result[mask == 255] = blended[mask == 255]
    return result


def apply_texture_overlay(img_bgr: np.ndarray, mask: np.ndarray, texture_bgr: np.ndarray) -> np.ndarray:
    """
    Apply a texture to the masked floor area (resized to image size).
    """
    h, w = img_bgr.shape[:2]
    texture_resized = cv2.resize(texture_bgr, (w, h))

    result = img_bgr.copy()
    result[mask == 255] = texture_resized[mask == 255]
    return result


def overlay_mask_outline(img_bgr: np.ndarray, mask: np.ndarray, colour=(0, 255, 255)) -> np.ndarray:
    """
    Draw the outline of the mask onto the image for visualization.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = img_bgr.copy()
    cv2.drawContours(outlined, contours, -1, colour, 2)
    return outlined
