###import turtle
from pathlib import Path

###import cv2
###import numpy as np
###from PIL import Image

IMAGE_PATH = "IMG_123.jpeg"

# Image settings
MAX_WIDTH = 450
MAX_HEIGHT = 600

# Dot-art settings
DOT_SPACING = 4
MIN_DOT_SIZE = 1
MAX_DOT_SIZE = 5

# Facial detail settings
EDGE_THRESHOLD = 65
FEATURE_DOT_SIZE = 2
FEATURE_DENSITY = 2

DRAW_SPEED = 0


def main():
    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((MAX_WIDTH, MAX_HEIGHT))

    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Smooth noise while retaining facial detail.
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    height, width = gray.shape

    # Detect edges around eyes, eyebrows, nostrils, lips,
    # and the face outline.
    edges = cv2.Canny(gray, 35, EDGE_THRESHOLD)

    # Slightly thicken edges so facial features stand out.
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    scale = min(700 / width, 850 / height)
    draw_width = width * scale
    draw_height = height * scale

    screen = turtle.Screen()
    screen.setup(
        width=int(draw_width + 60),
        height=int(draw_height + 80)
    )
    screen.title("Facial Expression Dot Art")
    screen.bgcolor("white")
    screen.tracer(0, 0)

    pen = turtle.Turtle(visible=False)
    pen.speed(DRAW_SPEED)
    pen.penup()
    pen.color("black")

    left = -draw_width / 2
    top = draw_height / 2

    # Pass 1: Draw facial details with small, dense dots.
    for y in range(0, height, FEATURE_DENSITY):
        for x in range(0, width, FEATURE_DENSITY):
            if edges[y, x] > 0:
                pen.goto(
                    left + x * scale,
                    top - y * scale
                )
                pen.dot(FEATURE_DOT_SIZE, "black")

        if y % 40 == 0:
            screen.update()

    # Pass 2: Add stippling for shadows and tonal depth.
    for y in range(0, height, DOT_SPACING):
        for x in range(0, width, DOT_SPACING):
            brightness = int(gray[y, x])
            darkness = 255 - brightness

            # Avoid filling bright areas with unnecessary dots.
            if darkness < 35:
                continue

            # Keep the dots smaller near strong facial edges.
            if edges[y, x] > 0:
                dot_size = 1
            else:
                dot_size = MIN_DOT_SIZE + (
                    darkness / 255
                ) * (MAX_DOT_SIZE - MIN_DOT_SIZE)

            pen.goto(
                left + x * scale,
                top - y * scale
            )
            pen.dot(max(1, int(dot_size)), "black")

        screen.update()

    screen.update()
    print("Facial detail dot sketch complete.")
    print("Close the Turtle window to exit.")
    screen.mainloop()


if __name__ == "__main__":
    main()