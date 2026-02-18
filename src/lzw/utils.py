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
    def compute_difference_image(image_array: np.ndarray) -> list[list[int]]:
        height = len(image_array)
        width = len(image_array[0])

        # Create empty 2D list to store differences
        diff_image: list[list[int]] = []
        for row in range(height):
            diff_image.append([0] * width)

        # First pixel stays as-is (top-left corner)
        diff_image[0][0] = int(image_array[0][0])

        # Row-wise differences for the first row (starting from 2nd pixel)
        # Each pixel = current pixel - previous pixel in same row
        for col in range(1, width):
            current_pixel = int(image_array[0][col])
            previous_pixel = int(image_array[0][col - 1])
            diff_image[0][col] = current_pixel - previous_pixel

        # Column-wise differences for the first column (starting from 2nd pixel)
        # Each pixel = current pixel - pixel above it
        for row in range(1, height):
            current_pixel = int(image_array[row][0])
            pixel_above = int(image_array[row - 1][0])
            diff_image[row][0] = current_pixel - pixel_above

        # Row-wise differences for remaining pixels
        # Each pixel = current pixel - previous pixel in same row
        for row in range(1, height):
            for col in range(1, width):
                current_pixel = int(image_array[row][col])
                previous_pixel = int(image_array[row][col - 1])
                diff_image[row][col] = current_pixel - previous_pixel

        return diff_image

    # Restore original image from difference image
    @staticmethod
    def restore_from_difference(diff_image: list[list[int]]) -> list[list[int]]:
        height = len(diff_image)
        width = len(diff_image[0])

        # Create empty 2D list to store restored image
        restored: list[list[int]] = []
        for row in range(height):
            restored.append([0] * width)

        # First pixel stays as-is
        restored[0][0] = diff_image[0][0]

        # Restore first row from row-wise differences
        # current pixel = previous pixel + difference
        for col in range(1, width):
            restored[0][col] = restored[0][col - 1] + diff_image[0][col]

        # Restore first column from column-wise differences
        # current pixel = pixel above + difference
        for row in range(1, height):
            restored[row][0] = restored[row - 1][0] + diff_image[row][0]

        # Restore remaining pixels from row-wise differences
        # current pixel = previous pixel + difference
        for row in range(1, height):
            for col in range(1, width):
                restored[row][col] = restored[row][col - 1] + diff_image[row][col]

        return restored

    # Convert difference image (2D list) to bytes
    # Differences range from -255 to 255
    # We shift by 255 so all values are positive (0 to 510)
    # Then store each value as 2 bytes (little-endian)
    @staticmethod
    def difference_to_bytes(diff_image: list[list[int]]) -> bytes:
        result = bytearray()

        for row in diff_image:
            for value in row:
                # Shift value: -255 becomes 0, 0 becomes 255, 255 becomes 510
                shifted_value = value + 255
                # Store as 2 bytes (little-endian: low byte first)
                low_byte = shifted_value & 0xFF
                high_byte = (shifted_value >> 8) & 0xFF
                result.append(low_byte)
                result.append(high_byte)

        return bytes(result)

    # Convert bytes back to difference image (2D list)
    @staticmethod
    def bytes_to_difference(data: bytes, height: int, width: int) -> list[list[int]]:
        # Create empty 2D list
        diff_image: list[list[int]] = []
        for row in range(height):
            diff_image.append([0] * width)

        # Read 2 bytes at a time and convert back to difference values
        byte_index = 0
        for row in range(height):
            for col in range(width):
                # Read 2 bytes (little-endian)
                low_byte = data[byte_index]
                high_byte = data[byte_index + 1]
                byte_index += 2

                # Combine bytes to get shifted value
                shifted_value = low_byte + (high_byte << 8)

                # Shift back: subtract 255 to get original difference
                diff_image[row][col] = shifted_value - 255

        return diff_image

    # Calculate entropy of data in bits per symbol
    # Entropy measures how much "randomness" or information is in the data
    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        if len(data) == 0:
            return 0.0

        # Count how many times each byte value (0-255) appears
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1

        # Calculate probability of each byte value
        total_bytes = len(data)
        entropy = 0.0

        for count in byte_counts:
            if count > 0:
                # Probability = count / total
                probability = count / total_bytes
                # Entropy formula: -sum(p * log2(p))
                import math

                entropy -= probability * math.log2(probability)

        return entropy

    # Calculate average code length (bits per symbol)
    @staticmethod
    def calculate_average_code_length(num_symbols: int, total_bits: int) -> float:
        if num_symbols == 0:
            return 0.0
        return total_bits / num_symbols
