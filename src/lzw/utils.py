# LZW Utils - Utility functions for LZW compression and decompression
import os


class LZWUtils:
    # Calculate the output file size after compression then compare it with
    # the original file size to determine the compression ratio
    @staticmethod
    def calculate_compression(original_file_path: str, compressed_file_path: str) -> str:

        original_size = os.path.getsize(original_file_path)
        compressed_size = os.path.getsize(compressed_file_path)

        if original_size == 0:
            return "0.0%"

        ratio = compressed_size / original_size * 100.0
        return (
            f"{ratio:.2f}% of original size ({compressed_size} bytes "
            f"compressed from {original_size} bytes)"
        )

    # Calculate the output file size after decompression then compare it with
    # the original file size to determine the decompression ratio
    @staticmethod
    def calculate_decompression(original_file_path: str, decompressed_file_path: str) -> str:
        original_size = os.path.getsize(original_file_path)
        decompressed_size = os.path.getsize(decompressed_file_path)

        if original_size == 0:
            return "0.0%"

        ratio = decompressed_size / original_size * 100.0
        return (
            f"{ratio:.2f}% of original size ({decompressed_size} bytes "
            f"decompressed from {original_size} bytes)"
        )

    # Check that the original file and the decompressed file are identical
    @staticmethod
    def verify_files(original_file_path: str, decompressed_file_path: str) -> bool:
        with open(original_file_path, "rb") as f1, open(decompressed_file_path, "rb") as f2:
            return f1.read() == f2.read()
