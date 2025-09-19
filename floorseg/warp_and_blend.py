# floorseg/warp_and_blend.py
import cv2
import numpy as np

def largest_contour_quad(mask: np.ndarray, padding=10):
    """
    Get a 4-point approximated quad of the largest mask contour that touches bottom.
    Returns points in order [[x0,y0],..[x3,y3]] or None.
    """
    # ensure binary 0/255
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest area contour
    c = max(contours, key=cv2.contourArea)
    # approximate polygon
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if approx has 4 points, accept; else approximate bounding quad by minAreaRect
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        pts = box.astype(np.int32)
    # order points (tl, tr, br, bl)
    def order_pts(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)
    quad = order_pts(pts)
    # optional padding outward along edges (small)
    # return float32
    return quad

def tile_texture_to_cover(texture: np.ndarray, target_w: int, target_h: int):
    th, tw = texture.shape[:2]
    if th >= target_h and tw >= target_w:
        return texture[:target_h, :target_w].copy()
    # tile
    rep_y = int(np.ceil(target_h / th))
    rep_x = int(np.ceil(target_w / tw))
    tiled = np.tile(texture, (rep_y, rep_x, 1))
    return tiled[:target_h, :target_w].copy()

def warp_texture_into_quad(texture_bgr: np.ndarray, quad_src_pts: np.ndarray, out_w: int, out_h: int):
    """
    Warp a tiled texture to the destination quad in an output canvas of size out_w x out_h.
    quad_src_pts: 4 points (tl,tr,br,bl) in destination image coordinates where texture corners should map to.
    We'll create a texture image sized to cover the bounding box of quad and warp it into place.
    """
    # bounding rect for quad
    xs = quad_src_pts[:, 0]; ys = quad_src_pts[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bw = max(2, x_max - x_min)
    bh = max(2, y_max - y_min)

    # make a tiled texture region sized to bounding box
    tex_region = tile_texture_to_cover(texture_bgr, bw, bh)

    # source corners (texture space)
    h_t, w_t = tex_region.shape[:2]
    src_pts = np.array([[0,0], [w_t-1,0], [w_t-1,h_t-1], [0,h_t-1]], dtype=np.float32)
    # dest corners = quad points shifted to local bounding box coords
    dst_pts = quad_src_pts - np.array([x_min, y_min], dtype=np.float32)

    # compute homography from texture -> destination polygon
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # warp the texture region to the bounding box
    warped = cv2.warpPerspective(tex_region, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # create full-size canvas and place warped region
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[y_min:y_min+bh, x_min:x_min+bw] = warped
    return canvas

def feather_mask(mask: np.ndarray, ksize=31):
    """Return a soft alpha mask (float32 0..1) by gaussian blurring the binary mask."""
    if mask.dtype != np.uint8:
        m = (mask > 0).astype(np.uint8) * 255
    else:
        m = mask.copy()
    # blur
    k = ksize if ksize % 2 == 1 else ksize+1
    blurred = cv2.GaussianBlur(m, (k,k), 0)
    alpha = blurred.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha

def apply_texture_perspective_and_blend(img_bgr: np.ndarray, mask: np.ndarray, texture_bgr: np.ndarray, seamless=True):
    """
    Full pipeline:
    - get quad from mask (largest contour)
    - warp tiled texture into that quad using homography
    - create mask region for blending
    - blend using seamlessClone or alpha blend with feathering
    Returns final image (BGR).
    """
    h, w = img_bgr.shape[:2]
    quad = largest_contour_quad(mask)
    if quad is None:
        return img_bgr.copy()

    # warp texture into quad
    warped_tex_canvas = warp_texture_into_quad(texture_bgr, quad, w, h)

    # create binary region where warped texture has nonzero content
    gray = cv2.cvtColor(warped_tex_canvas, cv2.COLOR_BGR2GRAY)
    region_mask = (gray > 5).astype(np.uint8) * 255

    if seamless:
        # find center of region for seamlessClone
        ys, xs = np.where(region_mask > 0)
        if len(xs) == 0:
            return img_bgr.copy()
        center = (int(xs.mean()), int(ys.mean()))
        # seamlessClone expects 3-channel src and dest, and 8-bit mask
        try:
            output = cv2.seamlessClone(warped_tex_canvas, img_bgr, region_mask, center, cv2.NORMAL_CLONE)
            return output
        except Exception:
            # fallback to alpha blend
            seamless = False

    # alpha blending fallback
    alpha = feather_mask(region_mask, ksize=41)[:,:,None]
    alpha = np.repeat(alpha, 3, axis=2)
    blended = (alpha * warped_tex_canvas.astype(np.float32) + (1-alpha) * img_bgr.astype(np.float32)).astype(np.uint8)
    return blended


# Alias for compatibility
warp_and_blend_texture = apply_texture_perspective_and_blend
