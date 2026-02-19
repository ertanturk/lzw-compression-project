# helper functions for LZW compression stuff
import os
import struct

import numpy as np
from PIL import Image


class LZWUtils:
    # compare file sizes before and after compression
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

    # calculate entropy for both files to see if info was preserved
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

    # check if original and decompressed files match
    @staticmethod
    def verify_files(original_file_path: str, decompressed_file_path: str) -> None:
        with open(original_file_path, "rb") as f1, open(decompressed_file_path, "rb") as f2:
            if f1.read() == f2.read():
                print(
                    "Verification successful: Original and decompressed files are identical." + "\n"
                )
            else:
                print("Verification failed: Original and decompressed files differ." + "\n")

    # same thing but for images - compares pixel values
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

    # open an image and return its pixel data as bytes + dimensions
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

    # instead of storing raw pixel values, store the differences between
    # neighboring pixels - this usually compresses better
    @staticmethod
    def compute_difference_image(image_array: np.ndarray) -> list[list[int]]:
        height = len(image_array)
        width = len(image_array[0])

        # empty 2D list for the differences
        diff_image: list[list[int]] = []
        for _row in range(height):
            diff_image.append([0] * width)

        # top-left corner stays the same
        diff_image[0][0] = int(image_array[0][0])

        # first row: difference = current - previous (left to right)
        for col in range(1, width):
            current_pixel = int(image_array[0][col])
            previous_pixel = int(image_array[0][col - 1])
            diff_image[0][col] = current_pixel - previous_pixel

        # first column: difference = current - the one above
        for row in range(1, height):
            current_pixel = int(image_array[row][0])
            pixel_above = int(image_array[row - 1][0])
            diff_image[row][0] = current_pixel - pixel_above

        # rest of the pixels: difference = current - previous in same row
        for row in range(1, height):
            for col in range(1, width):
                current_pixel = int(image_array[row][col])
                previous_pixel = int(image_array[row][col - 1])
                diff_image[row][col] = current_pixel - previous_pixel

        return diff_image

    # undo the differencing to get back the original pixel values
    @staticmethod
    def restore_from_difference(diff_image: list[list[int]]) -> list[list[int]]:
        height = len(diff_image)
        width = len(diff_image[0])

        # empty 2D list for restored image
        restored: list[list[int]] = []
        for _row in range(height):
            restored.append([0] * width)

        # top-left stays the same
        restored[0][0] = diff_image[0][0]

        # first row: add difference to previous pixel
        for col in range(1, width):
            restored[0][col] = restored[0][col - 1] + diff_image[0][col]

        # first column: add difference to pixel above
        for row in range(1, height):
            restored[row][0] = restored[row - 1][0] + diff_image[row][0]

        # rest: add difference to previous pixel in same row
        for row in range(1, height):
            for col in range(1, width):
                restored[row][col] = restored[row][col - 1] + diff_image[row][col]

        return restored

    # turn the 2D difference array into bytes
    # differences go from -255 to 255 so we shift by 255 to make them positive
    # then use 2 bytes per value (little-endian)
    @staticmethod
    def difference_to_bytes(diff_image: list[list[int]]) -> bytes:
        result = bytearray()

        for row in diff_image:
            for value in row:
                # shift so -255 becomes 0, 0 becomes 255, 255 becomes 510
                shifted_value = value + 255
                # split into 2 bytes
                low_byte = shifted_value & 0xFF
                high_byte = (shifted_value >> 8) & 0xFF
                result.append(low_byte)
                result.append(high_byte)

        return bytes(result)

    # reverse of difference_to_bytes - turn bytes back into 2D difference array
    @staticmethod
    def bytes_to_difference(data: bytes, height: int, width: int) -> list[list[int]]:
        # empty 2D list
        diff_image: list[list[int]] = []
        for _row in range(height):
            diff_image.append([0] * width)

        # read 2 bytes at a time and undo the shift
        byte_index = 0
        for row in range(height):
            for col in range(width):
                # read 2 bytes and combine
                low_byte = data[byte_index]
                high_byte = data[byte_index + 1]
                byte_index += 2

                # put bytes together
                shifted_value = low_byte + (high_byte << 8)

                # undo the shift to get the original difference
                diff_image[row][col] = shifted_value - 255

        return diff_image

    # calculate how many bits per symbol are needed (entropy)
    # lower entropy = more compressible
    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        if len(data) == 0:
            return 0.0

        # count each byte value (0-255)
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1

        # calculate probability and entropy
        total_bytes = len(data)
        entropy = 0.0

        for count in byte_counts:
            if count > 0:
                probability = count / total_bytes
                # entropy formula: -sum(p * log2(p))
                import math

                entropy -= probability * math.log2(probability)

        return entropy

    # average number of bits used per symbol
    @staticmethod
    def calculate_average_code_length(num_symbols: int, total_bits: int) -> float:
        if num_symbols == 0:
            return 0.0
        return total_bits / num_symbols

    # keep only one color channel by zeroing out the others
    @staticmethod
    def isolate_color_channel(image_array: np.ndarray, channel: str = "grayscale") -> np.ndarray:
        if channel not in ["red", "green", "blue", "grayscale"]:
            raise ValueError("Channel must be 'red', 'green', 'blue', or 'grayscale'")

        # make a copy so we dont mess up the original
        modified_image = np.copy(image_array)

        if channel == "red":
            modified_image[:, :, 1] = 0  # kill green
            modified_image[:, :, 2] = 0  # kill blue
        elif channel == "green":
            modified_image[:, :, 0] = 0  # kill red
            modified_image[:, :, 2] = 0  # kill blue
        elif channel == "blue":
            modified_image[:, :, 0] = 0  # kill red
            modified_image[:, :, 1] = 0  # kill green
        else:
            # 0.2989R + 0.5870G + 0.1140B (Grayscale conversion) information from GeeksforGeeks
            gray = (
                0.2989 * modified_image[:, :, 0]
                + 0.5870 * modified_image[:, :, 1]
                + 0.1140 * modified_image[:, :, 2]
            )
            modified_image[:, :, 0] = gray
            modified_image[:, :, 1] = gray
            modified_image[:, :, 2] = gray
        return modified_image

    # save a numpy array as an image file
    @staticmethod
    def save_image_from_array(image_array: np.ndarray, output_path: str) -> None:
        image = Image.fromarray(image_array)
        image.save(output_path)


class GUIUtils:
    # figure out if a file is an image or text based on the extension
    @staticmethod
    def get_file_type(file_path: str) -> str:
        image_extensions = [".png", ".bmp"]
        # grab the extension part
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        # see if it looks like an image
        if extension in image_extensions:
            return "image"
        else:
            return "text"

    # show the image with a specific color channel for the preview
    @staticmethod
    def apply_color_mode(file_path: str, color_mode: str) -> Image.Image:
        image = Image.open(file_path)
        # if default or not RGB just return as is
        if color_mode == "default":
            return image
        if image.mode != "RGB":
            return image
        # apply color isolation
        image_array = np.array(image)
        modified_array = LZWUtils.isolate_color_channel(image_array, color_mode)
        return Image.fromarray(modified_array.astype(np.uint8))

    # compress a text file and return the compressed data + stats
    @staticmethod
    def compress_text(file_path: str) -> tuple[bytes, dict[str, int | float]]:
        from .encoding import LZWEncoder

        # read the whole file
        with open(file_path, "rb") as f:
            data = f.read()

        # entropy of the original data
        original_entropy = LZWUtils.calculate_entropy(data)

        # run LZW on it
        encoder = LZWEncoder()
        encoded_data = encoder.encode(data)

        # get the file extension (like ".txt")
        _, extension = os.path.splitext(file_path)
        ext_bytes = extension.encode("utf-8")

        # build the .lzw file:
        # format: type(1 byte) + ext_length(1 byte) + extension + compressed_data
        lzw_data = bytearray()
        lzw_data.append(0)  # 0 = text file
        lzw_data.append(len(ext_bytes))  # extension length
        lzw_data.extend(ext_bytes)  # the extension
        lzw_data.extend(encoded_data)  # compressed stuff
        lzw_bytes = bytes(lzw_data)

        # gather stats
        original_size = os.path.getsize(file_path)
        compressed_size = len(lzw_bytes)
        num_symbols = len(data)
        total_bits = len(encoded_data) * 8
        avg_code_length = LZWUtils.calculate_average_code_length(num_symbols, total_bits)

        # calculate CR, CF, SS
        if compressed_size > 0:
            compression_ratio = compressed_size / original_size
            compression_factor = original_size / compressed_size
        else:
            compression_ratio = 0.0
            compression_factor = 0.0
        space_savings = (
            (original_size - compressed_size) / original_size if original_size > 0 else 0.0
        )

        stats: dict[str, int | float] = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "entropy": original_entropy,
            "avg_code_length": avg_code_length,
            "compression_ratio": compression_ratio,
            "compression_factor": compression_factor,
            "space_savings": space_savings,
        }

        return lzw_bytes, stats

    # compress an image file with color and method options
    @staticmethod
    def compress_image(
        file_path: str, color_mode: str = "default", method: str = "differences"
    ) -> tuple[bytes, dict[str, int | float]]:
        from .encoding import LZWEncoder

        # open the image
        image = Image.open(file_path)
        image_array = np.array(image)
        mode = image.mode

        # apply color mode if we need to (only for RGB)
        if color_mode != "default" and mode == "RGB":
            image_array = LZWUtils.isolate_color_channel(image_array, color_mode)
            # grayscale = single channel
            if color_mode == "grayscale":
                image_array = image_array[:, :, 0].astype(np.uint8)
                channels = 1
            else:
                channels = 3
        elif mode == "L":
            channels = 1
        else:
            channels = 3

        # get image size
        if channels == 1:
            height, width = image_array.shape
        else:
            height, width, _ = image_array.shape

        # pick the compression method
        if method == "differences":
            method_byte = 1
            if channels == 1:
                # single channel - compute differences directly
                diff_image = LZWUtils.compute_difference_image(image_array)
                data_bytes = LZWUtils.difference_to_bytes(diff_image)
            else:
                # RGB - compute differences for each channel separately
                all_bytes = bytearray()
                for ch in range(3):
                    channel_array = image_array[:, :, ch]
                    diff_image = LZWUtils.compute_difference_image(channel_array)
                    channel_bytes = LZWUtils.difference_to_bytes(diff_image)
                    all_bytes.extend(channel_bytes)
                data_bytes = bytes(all_bytes)
        else:
            # gray_levels = just use raw pixel values
            method_byte = 0
            data_bytes = image_array.flatten().tobytes()

        # entropy of original pixels
        original_bytes = image_array.flatten().tobytes()
        original_entropy = LZWUtils.calculate_entropy(original_bytes)

        # entropy of the data we're about to compress
        data_entropy = LZWUtils.calculate_entropy(data_bytes)

        # run LZW
        encoder = LZWEncoder()
        encoded_data = encoder.encode(data_bytes)

        # get file extension
        _, extension = os.path.splitext(file_path)
        ext_bytes = extension.encode("utf-8")

        # build the .lzw file:
        # format: type(1) + width(4) + height(4) + channels(1) + method(1)
        #         + ext_len(1) + extension + compressed_data
        lzw_data = bytearray()
        lzw_data.append(1)  # 1 = image file
        lzw_data.extend(struct.pack("<I", width))  # width (4 bytes)
        lzw_data.extend(struct.pack("<I", height))  # height (4 bytes)
        lzw_data.append(channels)  # channels (1 or 3)
        lzw_data.append(method_byte)  # method (0 or 1)
        lzw_data.append(len(ext_bytes))  # extension length
        lzw_data.extend(ext_bytes)  # extension string
        lzw_data.extend(encoded_data)  # compressed data
        lzw_bytes = bytes(lzw_data)

        # gather stats
        original_size = os.path.getsize(file_path)
        compressed_size = len(lzw_bytes)
        num_symbols = len(data_bytes)
        total_bits = len(encoded_data) * 8
        avg_code_length = LZWUtils.calculate_average_code_length(num_symbols, total_bits)

        # calculate CR, CF, SS
        if compressed_size > 0:
            compression_ratio = compressed_size / original_size
            compression_factor = original_size / compressed_size
        else:
            compression_ratio = 0.0
            compression_factor = 0.0
        space_savings = (
            (original_size - compressed_size) / original_size if original_size > 0 else 0.0
        )

        stats: dict[str, int | float] = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "original_entropy": original_entropy,
            "data_entropy": data_entropy,
            "avg_code_length": avg_code_length,
            "compression_ratio": compression_ratio,
            "compression_factor": compression_factor,
            "space_savings": space_savings,
        }

        return lzw_bytes, stats

    # decompress a .lzw file back to original data
    # returns (data, type, extension, info)
    @staticmethod
    def decompress_lzw(
        file_path: str,
    ) -> tuple[bytes | Image.Image, str, str, dict[str, int | str]]:
        from .decoding import LZWDecoder

        # read the whole .lzw file
        with open(file_path, "rb") as f:
            all_data = f.read()

        # first byte = file type (0=text, 1=image)
        file_type_byte = all_data[0]

        if file_type_byte == 0:
            # --- TEXT ---
            # read extension info
            ext_len = all_data[1]
            extension = all_data[2 : 2 + ext_len].decode("utf-8")
            # everything after the header is compressed data
            compressed_data = all_data[2 + ext_len :]

            # decompress it
            decoder = LZWDecoder()
            decompressed_bytes = decoder.decode(compressed_data)

            return decompressed_bytes, "text", extension, {}

        elif file_type_byte == 1:
            # --- IMAGE ---
            # parse the header
            offset = 1
            width = struct.unpack("<I", all_data[offset : offset + 4])[0]
            offset += 4
            height = struct.unpack("<I", all_data[offset : offset + 4])[0]
            offset += 4
            channels = all_data[offset]
            offset += 1
            method = all_data[offset]
            offset += 1
            ext_len = all_data[offset]
            offset += 1
            extension = all_data[offset : offset + ext_len].decode("utf-8")
            offset += ext_len
            # the rest after header is compressed data
            compressed_data = all_data[offset:]

            # decompress with LZW
            decoder = LZWDecoder()
            decoded_data = decoder.decode(compressed_data)

            # restore the image depending on which method was used
            if method == 1:
                # differences method
                if channels == 1:
                    diff_image = LZWUtils.bytes_to_difference(decoded_data, height, width)
                    restored = LZWUtils.restore_from_difference(diff_image)
                    image_array = np.array(restored, dtype=np.uint8)
                else:
                    # restore each RGB channel from its differences
                    bytes_per_channel = height * width * 2
                    restored_channels: list[list[list[int]]] = []
                    for ch in range(3):
                        start = ch * bytes_per_channel
                        end = start + bytes_per_channel
                        channel_bytes = decoded_data[start:end]
                        diff_image = LZWUtils.bytes_to_difference(channel_bytes, height, width)
                        restored_channel = LZWUtils.restore_from_difference(diff_image)
                        restored_channels.append(restored_channel)

                    # merge R, G, B channels back into one image
                    image_array = np.zeros((height, width, 3), dtype=np.uint8)
                    for row in range(height):
                        for col in range(width):
                            image_array[row][col][0] = restored_channels[0][row][col]
                            image_array[row][col][1] = restored_channels[1][row][col]
                            image_array[row][col][2] = restored_channels[2][row][col]
            else:
                # gray levels method - just raw pixels
                if channels == 1:
                    image_array = np.frombuffer(decoded_data, dtype=np.uint8).reshape(
                        (height, width)
                    )
                else:
                    image_array = np.frombuffer(decoded_data, dtype=np.uint8).reshape(
                        (height, width, 3)
                    )

            # make PIL image from array
            if channels == 1:
                image = Image.fromarray(image_array, mode="L")
            else:
                image = Image.fromarray(image_array, mode="RGB")

            # some extra info that might be useful
            extra_info: dict[str, int | str] = {
                "width": width,
                "height": height,
                "channels": channels,
                "method": "differences" if method == 1 else "gray_levels",
            }

            return image, "image", extension, extra_info

        else:
            raise ValueError(f"Unknown file type in .lzw file: {file_type_byte}")

    # write compressed bytes to a .lzw file
    @staticmethod
    def save_lzw_file(lzw_bytes: bytes, output_path: str) -> None:
        with open(output_path, "wb") as f:
            f.write(lzw_bytes)

    # save the decompressed data back to a file
    # text = write bytes, image = save with PIL
    @staticmethod
    def save_decompressed_file(data: bytes | Image.Image, file_type: str, output_path: str) -> None:
        if file_type == "text" and isinstance(data, bytes):
            with open(output_path, "wb") as f:
                f.write(data)
        elif file_type == "image" and isinstance(data, Image.Image):
            data.save(output_path)
