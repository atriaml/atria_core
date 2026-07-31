import cv2
import numpy as np
import pytesseract


def _detect_rotation_angle(image: np.ndarray) -> int:
    osd = pytesseract.image_to_osd(image)
    for line in osd.split("\n"):
        if "Rotate:" in line:
            angle = int(line.split(":")[1].strip())
            return angle
    return 0


def _correct_orientation(image: np.ndarray) -> np.ndarray:
    angle = _detect_rotation_angle(image)
    if angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Compute new bounding dimensions of the image
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image with new dimensions
        return cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        return image
