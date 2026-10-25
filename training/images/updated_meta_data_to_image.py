from pathlib import Path
# import json
import cv2
import math
# import copy
# import re

import pandas as pd


# bead_id = "combined_images"
bead_id = "bead_4"

BASE_DIR = Path(
    f"/Users/supriya/Documents/data/agentic_system/"
    f"multi-agent-manufacturing-inspection/training/images/"
    f"xiris_frames/temp_frames/temp_frames/"
    f"temp_frames_{bead_id}"
)

OUTPUT_DIR = (
    BASE_DIR
    / "melt_pool_detect"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_FILE = (
    BASE_DIR.parents[2]
    /
    "full_feature_multi_class.csv"
)

df = pd.read_csv(
    CSV_FILE,
    encoding="cp1252",
    keep_default_na=False,
    dtype={
        "frame_id": str
    }
)

required_columns = [
    "frame_id",
    "Cx",
    "Cy",
    "X",
    "Y",
    "Width",
    "Height",
    "Orientation",
    "FeretDiameter0",
    "FeretDiameter90",
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

scale = 10
processed = 0
failed = 0


for _, row in df.iterrows():

    frame_id = str(
        row["frame_id"]
    ).strip()

    image_file = (
        BASE_DIR
        /
        f"{frame_id}.png"
    )
    output_image_file = (
        OUTPUT_DIR
        /
        f"{frame_id}_overlay.png"
    )

    try:

        if not image_file.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_file}"
            )

        img = cv2.imread(
            str(image_file)
        )
        if img is None:

            raise ValueError(
                f"Could not decode image: "
                f"{image_file}"
            )

        Cx = float(
            row["Cx"]
        )

        Cy = float(
            row["Cy"]
        )

        X = float(
            row["X"]
        )
        Y = float(
            row["Y"]
        )
        Width = float(
            row["Width"]
        )
        Height = float(
            row["Height"]
        )
        Orientation = float(
            row["Orientation"]
        )
        FeretDiameter0 = float(
            row["FeretDiameter0"]
        )
        FeretDiameter90 = float(
            row["FeretDiameter90"]
        )

        x_px = int(
            round(
                X * scale
            )
        )
        y_px = int(
            round(
                Y * scale
            )
        )
        w_px = int(
            round(
                Width * scale
            )
        )
        h_px = int(
            round(
                Height * scale
            )
        )
        cx_px = int(
            round(
                Cx * scale
            )
        )
        cy_px = int(
            round(
                Cy * scale
            )
        )

        cv2.rectangle(
            img,
            (
                x_px,
                y_px
            ),
            (
                x_px + w_px,
                y_px + h_px
            ),
            (0, 255, 0),
            2
        )

        cv2.circle(
            img,
            (
                x_px,
                y_px
            ),
            4,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            img,
            f"X,Y=({X:g},{Y:g})",
            (
                x_px - 28,
                y_px - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.circle(
            img,
            (
                cx_px,
                cy_px
            ),
            5,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            img,
            f"C=({Cx:g},{Cy:g})",
            (
                x_px + w_px + 10,
                cy_px - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        theta = math.radians(
            Orientation
        )

        half_f0 = (
            FeretDiameter0
            /
            2
        )

        dx0 = (
            half_f0
            *
            math.cos(
                theta
            )
        )
        dy0 = (
            half_f0
            *
            math.sin(
                theta
            )
        )
        f0_start = (
            int(
                round(
                    cx_px - dx0
                )
            ),

            int(
                round(
                    cy_px - dy0
                )
            )
        )
        f0_end = (
            int(
                round(
                    cx_px + dx0
                )
            ),

            int(
                round(
                    cy_px + dy0
                )
            )
        )
        cv2.line(
            img,
            f0_start,
            f0_end,
            (255, 255, 0),
            2
        )
        cv2.putText(
            img,
            f"F0={FeretDiameter0:g}",
            (
                x_px + w_px + 22,
                y_px + 60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        theta90 = (
            theta
            +
            math.pi / 2
        )

        half_f90 = (
            FeretDiameter90
            /
            2
        )
        dx90 = (
            half_f90
            *
            math.cos(
                theta90
            )
        )
        dy90 = (
            half_f90
            *
            math.sin(
                theta90
            )
        )

        f90_start = (
            int(
                round(
                    cx_px - dx90
                )
            ),

            int(
                round(
                    cy_px - dy90
                )
            )
        )

        f90_end = (
            int(
                round(
                    cx_px + dx90
                )
            ),

            int(
                round(
                    cy_px + dy90
                )
            )
        )
        cv2.line(
            img,
            f90_start,
            f90_end,
            (255, 0, 255),
            2
        )
        cv2.putText(
            img,
            f"F90={FeretDiameter90:g}",
            (
                x_px + 80,
                y_px - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )
        width_y = (
            y_px
            +
            h_px
            +
            15
        )
        cv2.line(
            img,
            (
                x_px,
                width_y
            ),
            (
                x_px + w_px,
                width_y
            ),
            (0, 255, 0),
            2
        )
        cv2.line(
            img,
            (
                x_px,
                width_y - 4
            ),
            (
                x_px,
                width_y + 4
            ),
            (0, 255, 0),
            1
        )
        cv2.line(
            img,
            (
                x_px + w_px,
                width_y - 4
            ),
            (
                x_px + w_px,
                width_y + 4
            ),
            (0, 255, 0),
            1
        )
        cv2.putText(
            img,
            f"Width={Width:g}",
            (
                x_px + 15,
                width_y + 18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        label_y = (
            width_y
            +
            38
        )
        cv2.putText(
            img,
            "Melt Pool Region",
            (
                x_px,
                label_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f"Orientation={Orientation:g} deg",
            (
                x_px,
                label_y + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        height_x = (
            x_px
            -
            15
        )
        cv2.line(
            img,
            (
                height_x,
                y_px
            ),
            (
                height_x,
                y_px + h_px
            ),
            (0, 255, 0),
            2
        )
        cv2.line(
            img,
            (
                height_x - 4,
                y_px
            ),
            (
                height_x + 4,
                y_px
            ),
            (0, 255, 0),
            1
        )
        cv2.line(
            img,
            (
                height_x - 4,
                y_px + h_px
            ),
            (
                height_x + 4,
                y_px + h_px
            ),
            (0, 255, 0),
            1
        )
        cv2.putText(
            img,
            f"H={Height:g}",
            (
                height_x - 45,
                y_px + h_px // 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        success = cv2.imwrite(
            str(
                output_image_file
            ),
            img
        )
        if not success:
            raise IOError(
                f"Could not save image: "
                f"{output_image_file}"
            )
        processed += 1
    except Exception as e:
        failed += 1
        print(
            f"[ERROR] "
            f"{frame_id} | "
            f"{e}"
        )
print("\nProcessing complete.")
print(f"Processed: {processed}")
print(f"Failed:    {failed}")
print(f"Outputs:   {OUTPUT_DIR}")