from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import json
import pandas as pd


@dataclass
class Component:
    label: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]


def read_gray(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return frame


def finite_difference(fi: np.ndarray, fi1: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (4):
        f_D = f_(i+1) - f_i

    Use signed arithmetic so negative changes are preserved.
    """
    if fi.shape != fi1.shape:
        raise ValueError(f"Shape mismatch: {fi.shape} vs {fi1.shape}")

    return fi1.astype(np.int16) - fi.astype(np.int16)


def threshold_signed_difference(
    fD: np.ndarray,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper Eqs. (5) and (6):
        f+ = 1 where fD > T
        f- = 1 where fD < -T
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    f_plus = (fD > T).astype(np.uint8)
    f_minus = (fD < -T).astype(np.uint8)

    return f_plus, f_minus


def morphological_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Paper clustering stage:
    morphological closing connects nearby bright areas.

    kernel_size is the implementation parameter representing the
    chosen minimum-distance scale d_min.
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, List[Component]]:
    """
    Paper spatter-identification stage:
    connected-component analysis (CCA).

    This function only extracts components. Component acceptance
    criteria must be set from the paper's morphology/size concept
    and calibration on the target data.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    components: List[Component] = []

    for label in range(1, n_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]

        components.append(
            Component(
                label=label,
                area=area,
                centroid=(float(cx), float(cy)),
                bbox=(x, y, w, h),
            )
        )

    return labels, components


def build_component_mask(
    labels: np.ndarray,
    components: List[Component],
    min_area: int,
    max_area: int,
    max_width: int,
    max_height: int,
    max_aspect_ratio: float,
) -> Tuple[np.ndarray, List[Component]]:
    """
    Paper CCA stage:
    retain components with spatter-like size and morphology.

    Numeric limits must be calibrated for these thermal frames because
    the paper does not report universal values.
    """
    kept: List[Component] = []
    out = np.zeros(labels.shape, dtype=np.uint8)

    for component in components:
        x, y, width, height = component.bbox

        shorter_side = max(1, min(width, height))
        longer_side = max(width, height)
        aspect_ratio = longer_side / shorter_side

        size_ok = min_area <= component.area <= max_area
        dimensions_ok = width <= max_width and height <= max_height
        shape_ok = aspect_ratio <= max_aspect_ratio

        if size_ok and dimensions_ok and shape_ok:
            kept.append(component)
            out[labels == component.label] = 1

    return out, kept


def _run_pipeline(
    mask: np.ndarray,
    closing_kernel_size: int,
    min_area: int,
    max_area: int,
    max_width: int,
    max_height: int,
    max_aspect_ratio: float,
) -> Tuple[np.ndarray, List[Component], np.ndarray, np.ndarray]:
    """
    Run closing -> CCA -> size filter on a single binary mask.

    Returns (spatter_mask, kept_components, closed, labels).
    Extracted as a helper so the same pipeline can be applied to
    both f+ and f- without duplicating code.
    """
    closed = morphological_closing(mask, closing_kernel_size)
    labels, components = connected_components(closed)
    spatter_mask, kept_components = build_component_mask(
        labels=labels,
        components=components,
        min_area=min_area,
        max_area=max_area,
        max_width=max_width,
        max_height=max_height,
        max_aspect_ratio=max_aspect_ratio,
    )
    return spatter_mask, kept_components, closed, labels


def detect_potential_spatter(
    fi: np.ndarray,
    fi1: np.ndarray,
    T: int,
    closing_kernel_size: int,
    min_area: int,
    max_area: int,
    max_width: int,
    max_height: int,
    max_aspect_ratio: float,
    is_first_frame: bool = False,
) -> Tuple[
    np.ndarray,
    List[Component],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    Paper detection stage:
    finite difference -> threshold -> closing -> CCA -> size filtering.

    When is_first_frame=True (i=0), the paper requires the full pipeline
    to be run on both f+ and f-, because spatter in f0 is only detectable
    via f-. The two results are combined into a single unified mask and
    component list by re-running CCA on the OR of the two closed masks.

    Returns:
        spatter_mask      - unified binary mask of kept components
        kept_components   - kept components (centroids valid for refinement)
        f_plus            - raw f+ binary image (for saving)
        f_minus           - raw f- binary image (for saving)
        closed_plus       - morphological closing of f+ (for saving)
        labels            - label array matching kept_components
        closed_minus      - morphological closing of f- (for saving, None if
                            not first frame)
    """
    fD = finite_difference(fi, fi1)
    f_plus, f_minus = threshold_signed_difference(fD, T)

    if is_first_frame:
        # Paper: for i=0, run the full pipeline on f- as well.
        # Both closed masks are needed, so get them in single calls each.
        _, _, closed_plus, _ = _run_pipeline(
            f_plus,
            closing_kernel_size,
            min_area,
            max_area,
            max_width,
            max_height,
            max_aspect_ratio,
        )
        _, _, closed_minus, _ = _run_pipeline(
            f_minus,
            closing_kernel_size,
            min_area,
            max_area,
            max_width,
            max_height,
            max_aspect_ratio,
        )

        # OR the two closed masks and run CCA once on the result so that
        # every component has a label that indexes into a single, consistent
        # labels array. This is required because components_to_mask and
        # temporal_refinement both rely on that array.
        combined_closed = np.clip(
            closed_plus.astype(np.uint16) + closed_minus.astype(np.uint16),
            0,
            1,
        ).astype(np.uint8)

        # Single CCA call — result reused for both build_component_mask
        # and the returned labels array (fixes duplicate CCA call).
        unified_labels, unified_components = connected_components(combined_closed)
        spatter_mask, kept_components = build_component_mask(
            unified_labels,
            unified_components,
            min_area=min_area,
            max_area=max_area,
            max_width=max_width,
            max_height=max_height,
            max_aspect_ratio=max_aspect_ratio,
        )

        return (
            spatter_mask,
            kept_components,
            f_plus,
            f_minus,
            closed_plus,
            unified_labels,
            closed_minus,
        )

    # For i > 0: only f+ is processed (paper default).
    # Single _run_pipeline call covers closed_plus, labels, and the mask
    # (fixes the duplicate _run_pipeline call on f_plus).
    spatter_mask, kept_components, closed_plus, labels = _run_pipeline(
        f_plus,
        closing_kernel_size,
        min_area,
        max_area,
        max_width,
        max_height,
        max_aspect_ratio,
    )

    return (
        spatter_mask,
        kept_components,
        f_plus,
        f_minus,
        closed_plus,
        labels,
        None,  # closed_minus not applicable
    )


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def temporal_refinement(
    current_components: List[Component],
    neighbor_components: List[Component],
    R: float,
) -> List[Component]:
    """
    Paper refinement concept:
    if a potential component has a barycenter within translation radius R
    of a potential component in a neighboring frame, treat it as a false
    positive caused by a persistent specular reflection.
    """
    refined: List[Component] = []

    for current in current_components:
        persistent = any(
            euclidean_distance(current.centroid, neighbor.centroid) < R
            for neighbor in neighbor_components
        )

        if not persistent:
            refined.append(current)

    return refined


def components_to_mask(
    labels: np.ndarray,
    components: List[Component],
) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.uint8)
    for c in components:
        out[labels == c.label] = 1
    return out


def summarize_components(
    components: List[Component],
    labels: np.ndarray,
    gray_image: np.ndarray,
) -> dict:
    """
    Extracts spatter blob features from refined components.
    Includes count, area statistics, centroids, and local contrast.
    Gonzalez & Woods (2018), Franke et al. (2025).
    """
    if not components:
        return {
            "spatter_count":         0,
            "largest_spatter_area_px": 0,
            "total_spatter_area_px": 0,
            "mean_blob_area_px":     0.0,
            "spatter_blobs":         [],
        }

    blobs = []
    for c in components:
        x, y, w, h = c.bbox
        pad = 5
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(gray_image.shape[1], x + w + pad)
        y2 = min(gray_image.shape[0], y + h + pad)

        neighborhood    = gray_image[y1:y2, x1:x2].astype(float)
        blob_mask_local = (labels[y1:y2, x1:x2] == c.label)

        blob_mean = float(neighborhood[blob_mask_local].mean()) \
                    if blob_mask_local.sum() > 0 else 0.0
        bg_pixels = neighborhood[~blob_mask_local]
        bg_mean   = float(bg_pixels.mean()) if len(bg_pixels) > 0 else 0.0

        blobs.append({
            "centroid_xy":      [round(c.centroid[0], 2), round(c.centroid[1], 2)],
            "area_px":          c.area,
            "local_contrast_v": round(blob_mean - bg_mean, 2),
        })

    areas = [b["area_px"] for b in blobs]

    return {
        "spatter_count":           len(blobs),
        "largest_spatter_area_px": max(areas),
        "total_spatter_area_px":   int(sum(areas)),
        "mean_blob_area_px":       round(sum(areas) / len(areas), 2),
        "spatter_blobs":           blobs,
    }


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output = (mask.astype(np.uint8) * 255)

    success = cv2.imwrite(str(path), output)

    if not success:
        raise RuntimeError(f"Failed to save image: {path}")

    print(f"Saved: {path}")

def save_refined_overlay(
    original_frame_path: Path,
    refined_mask: np.ndarray,
    output_path: Path,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.65,
) -> None:
    """
    Overlay the final refined spatter mask on the matching original frame.
    """
    original = cv2.imread(
        str(original_frame_path),
        cv2.IMREAD_COLOR,
    )

    if original is None:
        raise FileNotFoundError(
            f"Cannot read original frame: {original_frame_path}"
        )

    binary_mask = refined_mask.astype(np.uint8) > 0

    if original.shape[:2] != binary_mask.shape:
        raise ValueError(
            "Original frame and refined mask dimensions do not match: "
            f"{original.shape[:2]} vs {binary_mask.shape}"
        )

    overlay = original.copy()

    color_layer = np.zeros_like(original)
    color_layer[:] = color

    blended = cv2.addWeighted(
        original,
        1.0 - alpha,
        color_layer,
        alpha,
        0,
    )

    overlay[binary_mask] = blended[binary_mask]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), overlay)

    if not success:
        raise RuntimeError(
            f"Failed to save overlay: {output_path}"
        )

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    bead_id = "bead_1"
    # bead_id = "bead_2"
    # bead_id = "bead_3"
    # bead_id = "bead_4"
    # bead_id = "bead_5"
    # bead_id = "bead_6"

    FRAME_FOLDER = Path(
        f"/Users/supriya/Documents/data/agentic_system/"
        f"multi-agent-manufacturing-inspection/training/images/"
        f"xiris_frames/temp_frames/temp_frames/temp_frames_{bead_id}"
    )

    MELT_POOL_OVERLAY_FOLDER = (FRAME_FOLDER
        / "melt_pool_detect"
    )

    OUTPUT_FOLDER = FRAME_FOLDER / "paper_spatter_detection"

    F_PLUS_FOLDER = OUTPUT_FOLDER / "f_plus"
    F_MINUS_FOLDER = OUTPUT_FOLDER / "f_minus"
    CLOSED_PLUS_FOLDER = OUTPUT_FOLDER / "closed_plus"
    CLOSED_MINUS_FOLDER = OUTPUT_FOLDER / "closed_minus"
    POTENTIAL_FOLDER = OUTPUT_FOLDER / "potential_spatter"
    REFINED_FOLDER = OUTPUT_FOLDER / "refined_spatter"
    REFINED_OVERLAY_FOLDER = (
        OUTPUT_FOLDER / "refined_spatter_overlays"
)

    for folder in (
        F_PLUS_FOLDER,
        F_MINUS_FOLDER,
        CLOSED_PLUS_FOLDER,
        CLOSED_MINUS_FOLDER,
        POTENTIAL_FOLDER,
        REFINED_FOLDER,
        REFINED_OVERLAY_FOLDER,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    # ── Melt pool metadata ─────────────────────────────────────────────────
    # MELT_POOL_CSV         = FRAME_FOLDER / "melt_pool_metadata.csv"
    MELT_POOL_CSV = Path(
    "/Users/supriya/Documents/data/agentic_system/"
    "multi-agent-manufacturing-inspection/training/images/"
    "xiris_frames/full_feature_multi_class.csv"
)
    COMBINED_METADATA_DIR = FRAME_FOLDER / "merged_metadata"
    COMBINED_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # df_meltpool = pd.read_csv(MELT_POOL_CSV)
    try:
        df_meltpool = pd.read_csv(MELT_POOL_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        df_meltpool = pd.read_csv(MELT_POOL_CSV, encoding="mac_roman", dtype={"frame_id": str})

    df_meltpool.columns   = df_meltpool.columns.str.strip()
    df_meltpool["frame_id"] = df_meltpool["frame_id"].astype(str).str.strip().str.replace("'", "")

    T = 20
    CLOSING_KERNEL_SIZE = 3
    MIN_AREA = 2
    MAX_AREA = 150
    R = 5.0
    MAX_WIDTH = 20
    MAX_HEIGHT = 20
    MAX_ASPECT_RATIO = 3.0

    image_files = sorted(
        p
        for p in FRAME_FOLDER.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and "_overlay" not in p.name
    )

    print(f"Frames found: {len(image_files)}")
    print(f"Output folder: {OUTPUT_FOLDER}")

    if len(image_files) < 2:
        raise RuntimeError("At least two consecutive frames are required.")

    detections = []

    for i in range(len(image_files) - 1):
        frame_i_path = image_files[i]
        frame_i1_path = image_files[i + 1]

        fi = read_gray(frame_i_path)
        fi1 = read_gray(frame_i1_path)

        (
            mask,
            components,
            f_plus,
            f_minus,
            closed_plus,
            labels,
            closed_minus,       
        ) = detect_potential_spatter(
            fi=fi,
            fi1=fi1,
            T=T,
            closing_kernel_size=CLOSING_KERNEL_SIZE,
            min_area=MIN_AREA,
            max_area=MAX_AREA,
            max_width=MAX_WIDTH,
            max_height=MAX_HEIGHT,
            max_aspect_ratio=MAX_ASPECT_RATIO,
            is_first_frame=(i == 0),
        )

        detections.append(
            {
                "frame_id": frame_i1_path.stem,
                "mask": mask,
                "labels": labels,
                "components": components,
                "f_plus": f_plus,
                "f_minus": f_minus,
                "closed_plus": closed_plus,
                "closed_minus": closed_minus,   # stored for saving below
            }
        )

    for i, detection in enumerate(detections):
        current_components = detection["components"]

        previous_components = (
            detections[i - 1]["components"] if i > 0 else []
        )
        next_components = (
            detections[i + 1]["components"]
            if i < len(detections) - 1
            else []
        )

        refined_components = current_components

        if previous_components:
            refined_components = temporal_refinement(
                current_components=refined_components,
                neighbor_components=previous_components,
                R=R,
            )

        if next_components:
            refined_components = temporal_refinement(
                current_components=refined_components,
                neighbor_components=next_components,
                R=R,
            )

        refined_mask = components_to_mask(
            labels=detection["labels"],
            components=refined_components,
        )

        frame_id  = detection["frame_id"]
        gray_fi1  = read_gray(image_files[i + 1])   # grayscale of current frame

        summary = summarize_components(
            components=refined_components,
            labels=detection["labels"],
            gray_image=gray_fi1,
        )

        print(
            f"{frame_id} | "
            f"refined_count={summary['spatter_count']} | "
            f"largest={summary['largest_spatter_area_px']} | "
            f"total={summary['total_spatter_area_px']}"
        )

        save_mask(
            detection["f_plus"],
            F_PLUS_FOLDER / f"{frame_id}_f_plus.png",
        )
        save_mask(
            detection["f_minus"],
            F_MINUS_FOLDER / f"{frame_id}_f_minus.png",
        )
        save_mask(
            detection["closed_plus"],
            CLOSED_PLUS_FOLDER / f"{frame_id}_closed_plus.png",
        )

        # Save closed_minus only for the first frame pair (i == 0).
        if detection["closed_minus"] is not None:
            save_mask(
                detection["closed_minus"],
                CLOSED_MINUS_FOLDER / f"{frame_id}_closed_minus.png",
            )

        save_mask(
            detection["mask"],
            POTENTIAL_FOLDER / f"{frame_id}_potential_spatter.png",
        )
        save_mask(
            refined_mask,
            REFINED_FOLDER / f"{frame_id}_refined_spatter.png",
        )

        original_frame_path = (
        MELT_POOL_OVERLAY_FOLDER
            / f"{frame_id}_overlay.png"
        )

        if not original_frame_path.exists():
            raise FileNotFoundError(
                f"Melt-pool overlay image not found: "
                f"{original_frame_path}"
        )

        save_refined_overlay(
            original_frame_path=original_frame_path,
            refined_mask=refined_mask,
            output_path=(
                REFINED_OVERLAY_FOLDER
                / f"{frame_id}_refined_spatter_overlay.png"
        ),
        color=(255, 255, 255),
        alpha=0.9,
        )

        # ── Merge melt pool + spatter into combined metadata JSON ──────────
        search_key = frame_id[-6:]
        melt_row   = df_meltpool[df_meltpool["frame_id"].str.endswith(search_key)]

        if melt_row.empty:
            print(f"Warning: No melt pool match for frame {frame_id}")
            melt_pool_dict = {}
        else:
            MELT_POOL_COLS = [
                "Area", "Cx", "Cy", "X", "Y", "Width", "Height",
                "Orientation", "FeretDiameter0", "FeretDiameter90"
            ]
            melt_pool_dict = melt_row.iloc[0][MELT_POOL_COLS].to_dict()

        combined = {
            "frame_id":         frame_id,
            "melt_pool_region": melt_pool_dict,
            "spatter_features": summary
        }

        out_path = COMBINED_METADATA_DIR / f"{frame_id}_metadata.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)

    print("\nProcessing complete.")