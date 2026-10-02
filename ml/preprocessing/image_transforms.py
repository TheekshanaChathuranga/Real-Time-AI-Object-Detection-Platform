"""Image preprocessing, validation, and transformations for the ML inference pipeline."""

from typing import Tuple, Union
import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocesses raw images, video frames, and byte buffers for inference."""

    @staticmethod
    def load_image(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Load and validate an image from file path, raw bytes, or existing numpy array.
        
        Returns:
            np.ndarray: BGR image array suitable for OpenCV / YOLO.
        Raises:
            ValueError: If input is corrupt or cannot be decoded.
        """
        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty.")
            return image_input

        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes. Unsupported or corrupted file.")
            return img

        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
            return img

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize and pad image while maintaining aspect ratio (letterboxing).
        
        Returns:
            Tuple: (padded_image, scale_ratio, (pad_left, pad_top))
        """
        shape = image.shape[:2]  # [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2  # divide padding into 2 sides
        dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, r, (left, top)
