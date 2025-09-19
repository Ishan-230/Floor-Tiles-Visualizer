"""
segmentation.py
Floor segmentation utilities: heuristic and model-based.
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ----------------------------
# Heuristic Segmentation
# ----------------------------
def heuristic_floor_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple heuristic segmentation using GrabCut.
    Assumes the lower half of the image is floor.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Assume bottom half is floor initially
    rect = (0, h // 2, w, h // 2)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    return mask2 * 255

# ----------------------------
# Model-based Segmentation
# ----------------------------
def load_segmentation_model(device: str = "cpu"):
    """
    Load DeepLabV3 ResNet50 model pre-trained on COCO-stuff.
    """
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model

def model_floor_mask(img_bgr: np.ndarray, device: str = "cpu", model=None) -> np.ndarray:
    """
    Use DeepLabV3 to predict a floor mask.
    COCO-stuff class IDs 9 ('floor') and 15 ('earth/ground') are used.
    """
    h, w = img_bgr.shape[:2]
    if model is None:
        model = load_segmentation_model(device)

    # Preprocess image
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Run segmentation
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    pred = output.argmax(0).byte().cpu().numpy()

    # Extract floor mask (COCO-stuff class IDs 9, 15)
    floor_mask = np.isin(pred, [9, 15]).astype(np.uint8) * 255

    # Resize back to original
    floor_mask = cv2.resize(floor_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Debug: Save mask for inspection
    # cv2.imwrite('floor_mask.png', floor_mask)

    # Validate mask
    if np.count_nonzero(floor_mask) == 0:
        print("Warning: Floor mask is empty. Consider using a custom mask or adjusting class IDs.")
    
    return floor_mask

# ----------------------------
# Utility: Load external mask
# ----------------------------
def load_binary_mask(path, target_shape=None) -> np.ndarray:
    """
    Load a binary mask from file path OR file-like object (Streamlit uploader).
    Optionally resize to target_shape.
    """
    mask = None

    # Case 1: path is a string/Path (normal file on disk)
    if isinstance(path, (str, bytes, os.PathLike)):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # Case 2: path is a file-like object (e.g. Streamlit upload)
    else:
        try:
            file_bytes = np.asarray(bytearray(path.read()), dtype=np.uint8)
            mask = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        finally:
            path.seek(0)  # reset pointer for Streamlit

    if mask is None:
        raise ValueError("Mask file not found or unreadable")

    # Ensure binary (0/255)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Resize if needed
    if target_shape is not None:
        mask_bin = cv2.resize(mask_bin, (target_shape[1], target_shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    return mask_bin


# ----------------------------
# Unified Entry Point
# ----------------------------
def get_floor_mask(
    img_bgr: np.ndarray,
    method: str = "heuristic",
    device: str = "cpu",
    model=None
) -> np.ndarray:
    """
    Unified entry point for floor mask generation.
    method: "heuristic" or "model"
    - If method="heuristic": runs GrabCut heuristic
    - If method="model":
        * If `model` (FloorSegModel) is provided, uses it
        * Else loads a temporary DeepLabV3 model
    """
    if method == "heuristic":
        return heuristic_floor_mask_bgr(img_bgr)
    elif method == "model":
        if model is not None:
            return model.predict(img_bgr)
        else:
            return model_floor_mask(img_bgr, device=device)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

# ----------------------------
# Wrapper Class for Compatibility
# ----------------------------
class FloorSegModel:
    """Wrapper class for DeepLabV3-based floor segmentation."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = load_segmentation_model(device)

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Predict floor mask using DeepLabV3.
        """
        return model_floor_mask(img_bgr, device=self.device, model=self.model)
    
#------------------------------------
# SAM (Meta's Segment Anything Model)
#------------------------------------
class SAMRefiner:
    def __init__(self, model_type="vit_b", checkpoint="sam_vit_b.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAM] Initializing with {model_type} on {self.device}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def refine(self, image_bgr: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
        """
        Refine a rough floor mask using SAM.
        Returns a binary mask.
        """
        import cv2, numpy as np

        # Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate masks
        masks = self.mask_generator.generate(image_rgb)
        if not masks:
            print("[SAM] No masks generated, returning rough mask")
            return rough_mask

        # Find SAM mask overlapping most with rough mask
        best_iou, best_mask = 0, None
        rough_bool = rough_mask > 0
        for m in masks:
            sam_mask = m["segmentation"]
            intersection = np.logical_and(rough_bool, sam_mask).sum()
            union = np.logical_or(rough_bool, sam_mask).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_mask = iou, sam_mask

        if best_mask is None:
            print("[SAM] No overlapping mask found, returning rough mask")
            return rough_mask.astype(np.uint8) * 255

        print(f"[SAM] Refinement successful, IoU={best_iou:.3f}")
        return (best_mask.astype(np.uint8)) * 255