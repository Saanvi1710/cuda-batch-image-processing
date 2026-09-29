
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("data/input")
NUM_IMAGES = 100
WIDTH = 1024
HEIGHT = 1024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

for i in range(NUM_IMAGES):
    image = np.random.randint(
        0, 256,
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    # Add geometric structures
    for _ in range(20):
        x1 = np.random.randint(0, WIDTH)
        y1 = np.random.randint(0, HEIGHT)
        x2 = np.random.randint(0, WIDTH)
        y2 = np.random.randint(0, HEIGHT)

        color = tuple(
            int(x) for x in np.random.randint(0, 256, 3)
        )

        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=np.random.randint(1, 5)
        )

    for _ in range(10):
        center = (
            np.random.randint(0, WIDTH),
            np.random.randint(0, HEIGHT)
        )

        radius = np.random.randint(10, 150)

        color = tuple(
            int(x) for x in np.random.randint(0, 256, 3)
        )

        cv2.circle(
            image,
            center,
            radius,
            color,
            thickness=np.random.randint(1, 5)
        )

    output_path = OUTPUT_DIR / f"image_{i:03d}.png"
    cv2.imwrite(str(output_path), image)

print(f"Generated {NUM_IMAGES} images in {OUTPUT_DIR}")
