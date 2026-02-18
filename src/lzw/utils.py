# LZW Utils - Utility functions for LZW compression and decompression
import os

import numpy as np
from PIL import Image


class LZWUtils:
    # Calculate the output file size after compression then compare it with
    # the original file size to determine the compression ratio
    @staticmethod
    def calculate_compression(original_file_path: str, compressed_file_path: str) -> str:

        original_size = os.path.getsize(original_file_path)
        compressed_size = os.path.getsize(compressed_file_path)

        if original_size == 0:
            return "0.0%"

        compression_ratio = compressed_size / original_size
        compression_factor = (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        )
        space_savings = (original_size - compressed_size) / original_size

        return (
            f"Compression Ratio: {compression_ratio:.2f}, "
            f"Compression Factor: {compression_factor:.2f}, "
            f"Space Savings: {space_savings:.2%}"
        )

    # Calculate the entropy of the original file and
    # the decompressed file to determine how much information is preserved
    @staticmethod
    def calculate_decompression(compressed_file_path: str, decompressed_file_path: str) -> str:
        def calculate_entropy(file_path: str) -> float:
            with open(file_path, "rb") as f:
                data = f.read()
            if not data:
                return 0.0
            byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
            probabilities = byte_counts / len(data)
            entropy = -np.sum(
                probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])
            )
            return entropy

        original_entropy = calculate_entropy(compressed_file_path)
        decompressed_entropy = calculate_entropy(decompressed_file_path)

        return (
            f"Original Entropy: {original_entropy:.4f} bits/byte, "
            f"Decompressed Entropy: {decompressed_entropy:.4f} bits/byte"
        )

    # Check that the original file and the decompressed file are identical
    @staticmethod
    def verify_files(original_file_path: str, decompressed_file_path: str) -> None:
        with open(original_file_path, "rb") as f1, open(decompressed_file_path, "rb") as f2:
            if f1.read() == f2.read():
                print(
                    "Verification successful: Original and decompressed files are identical." + "\n"
                )
            else:
                print("Verification failed: Original and decompressed files differ." + "\n")

    # Check that the original image and the decompressed image have identical pixel data
    @staticmethod
    def verify_image_files(original_file_path: str, decompressed_file_path: str) -> None:
        original_image = Image.open(original_file_path)
        decompressed_image = Image.open(decompressed_file_path)
        original_array = np.array(original_image)
        decompressed_array = np.array(decompressed_image)
        if np.array_equal(original_array, decompressed_array):
            print("Verification successful: Original and decompressed images are identical." + "\n")
        else:
            print("Verification failed: Original and decompressed images differ." + "\n")

    # Open image file, based on channel count, extract pixel data and
    # return flattened bytes with width and height for image reconstruction
    @staticmethod
    def open_image_file(image_file_path: str) -> tuple[bytes, int, int, int]:
        image = Image.open(image_file_path)
        mode = image.mode
        if mode == "L":  # Grayscale
            array = np.array(image)
            height, width = array.shape
            flatten_array = array.flatten().tobytes()
            return flatten_array, width, height, 1
        elif mode == "RGB":  # Color
            array = np.array(image)
            height, width, _ = array.shape
            flatten_array = array.flatten().tobytes()
            return flatten_array, width, height, 3
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

    # Compute difference image for better compression
    # Row-wise differences for each row (starting from 2nd pixel)
    # Column-wise differences for first column (starting from 2nd pixel)
    @staticmethod
    def compute_difference_image(image_array: np.ndarray) -> np.ndarray:
        height, width = image_array.shape
        # Use int16 to handle negative differences (-255 to 255)
        diff_image = np.zeros((height, width), dtype=np.int16)

        # First pixel stays as-is (top-left corner)
        diff_image[0, 0] = image_array[0, 0]

        # Row-wise differences for the first row (starting from 2nd pixel)
        for col in range(1, width):
            diff_image[0, col] = int(image_array[0, col]) - int(image_array[0, col - 1])

        # Column-wise differences for the first column (starting from 2nd pixel)
        for row in range(1, height):
            diff_image[row, 0] = int(image_array[row, 0]) - int(image_array[row - 1, 0])

        # Row-wise differences for remaining pixels
        for row in range(1, height):
            for col in range(1, width):
                diff_image[row, col] = int(image_array[row, col]) - int(image_array[row, col - 1])

        return diff_image

    # Restore original image from difference image
    @staticmethod
    def restore_from_difference(diff_image: np.ndarray) -> np.ndarray:
        height, width = diff_image.shape
        restored = np.zeros((height, width), dtype=np.uint8)

        # First pixel
        restored[0, 0] = diff_image[0, 0]

        # Restore first row from row-wise differences
        for col in range(1, width):
            restored[0, col] = restored[0, col - 1] + diff_image[0, col]

        # Restore first column from column-wise differences
        for row in range(1, height):
            restored[row, 0] = restored[row - 1, 0] + diff_image[row, 0]

        # Restore remaining pixels from row-wise differences
        for row in range(1, height):
            for col in range(1, width):
                restored[row, col] = restored[row, col - 1] + diff_image[row, col]

        return restored

    # Convert difference image to bytes (shift to unsigned range)
    # Differences range from -255 to 255, shift by 255 to get 0-510
    @staticmethod
    def difference_to_bytes(diff_image: np.ndarray) -> bytes:
        # Shift to unsigned range: -255 to 255 -> 0 to 510
        shifted = (diff_image.astype(np.int16) + 255).astype(np.uint16)
        return shifted.tobytes()

    # Convert bytes back to difference image
    @staticmethod
    def bytes_to_difference(data: bytes, height: int, width: int) -> np.ndarray:
        shifted = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
        diff_image = shifted.astype(np.int16) - 255
        return diff_image

    # Calculate entropy of data in bits per symbol
    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(data)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    # Calculate average code length
    @staticmethod
    def calculate_average_code_length(num_codes: int, total_bits: int) -> float:
        if num_codes == 0:
            return 0.0
        return total_bits / num_codes
