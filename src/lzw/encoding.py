import os
import struct

import numpy as np
from PIL import Image

from . import utils


class LZWEncoder:
    # Constructor initializes the encoder
    # with the base dictionary (256 single-byte entries) and sets the initial code size
    def __init__(self) -> None:
        self.dictionary = {bytes([i]): i for i in range(256)}
        self.dictionary_limit = 4096  # Maximum number of entries in the dictionary (12 bits)
        self.reset()

    # Resets the encoder to its initial state
    def reset(self) -> None:
        self.dictionary: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.next_code: int = 256
        self.code_size: int = 9  # Initial code size (9 bits to accommodate 256 single-byte entries)
        self.current_string: bytes = b""
        self.bit_buffer: int = 0
        self.bit_count: int = 0

    # Encodes the chunk of data using the LZW algorithm
    def encode(self, data: bytes) -> bytes:
        self.reset()
        result = bytearray()

        # Process each byte in the input data and build the dictionary dynamically
        for byte in data:
            current_byte = bytes([byte])
            combined_string = self.current_string + current_byte

            if combined_string in self.dictionary:
                self.current_string = combined_string
            else:
                # Output the code for the current string
                code = self.dictionary[self.current_string]
                result.extend(self.pack_code(code))

                # Add the combined string to the dictionary if there's room
                if self.next_code < self.dictionary_limit:
                    self.dictionary[combined_string] = self.next_code
                    self.next_code += 1
                    self.calculate_code_size()

                self.current_string = current_byte

        # Output the code for the last string if it exists
        if self.current_string:
            code = self.dictionary[self.current_string]
            result.extend(self.pack_code(code))

        # Flush any remaining bits in the buffer
        result.extend(self.flush())
        return bytes(result)

    # Increases the code size when the dictionary grows beyond the current bit width limit
    def calculate_code_size(self) -> int:
        if self.next_code == (1 << self.code_size) and self.code_size < 12:
            self.code_size += 1
        return self.code_size

    # Packs the code into the bit buffer and returns bytes when enough bits are accumulated
    def pack_code(self, code: int) -> bytes:
        self.bit_buffer |= code << self.bit_count
        self.bit_count += self.code_size
        result = bytearray()
        while self.bit_count >= 8:
            result.append(self.bit_buffer & 0xFF)
            self.bit_buffer >>= 8
            self.bit_count -= 8
        return result

    # Flushes the remaining bits in the buffer and returns any remaining bytes
    def flush(self) -> bytes:
        result = bytearray()

        while self.bit_count > 0:
            result.append(self.bit_buffer & 0xFF)
            self.bit_buffer >>= 8
            self.bit_count -= 8

        return bytes(result)

    # Reads a file, encodes its contents using LZW, and writes the result to the output file
    def encode_file(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as f:
            data = f.read()
        encoded_data = self.encode(data)
        with open(output_file_path, "wb") as f:
            f.write(encoded_data)
        print(f'Encoded "{input_file_path}" to "{output_file_path}" successfully.')
        print(f"Calculated code length: {len(encoded_data)} bytes")
        print(utils.LZWUtils.calculate_compression(input_file_path, output_file_path) + "\n")

    # Open an image file, compute difference image, encode using LZW,
    # and write the compressed data with statistics
    def encode_image_file(self, image_file_path: str, output_file_path: str) -> None:
        # Read the image file
        image = Image.open(image_file_path)
        mode = image.mode

        if mode == "L":  # Grayscale
            image_array = np.array(image)
            height, width = image_array.shape
            channels = 1

            # Compute difference image
            diff_image = utils.LZWUtils.compute_difference_image(image_array)
            diff_bytes = utils.LZWUtils.difference_to_bytes(diff_image)

        elif mode == "RGB":  # Color - process each channel
            image_array = np.array(image)
            height, width, _ = image_array.shape
            channels = 3

            # Compute difference image for each channel
            diff_channels: list[np.ndarray] = []
            for c in range(3):
                diff_channel = utils.LZWUtils.compute_difference_image(image_array[:, :, c])
                diff_channels.append(diff_channel)
            diff_image = np.stack(diff_channels, axis=-1)
            diff_bytes = diff_image.astype(np.int16)
            # Shift and convert to bytes
            shifted = (diff_bytes + 255).astype(np.uint16)
            diff_bytes = shifted.tobytes()
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

        # Calculate entropy of original image
        original_bytes, _, _, _ = utils.LZWUtils.open_image_file(image_file_path)
        original_entropy = utils.LZWUtils.calculate_entropy(original_bytes)

        # Calculate entropy of difference image
        diff_entropy = utils.LZWUtils.calculate_entropy(diff_bytes)

        # Encode difference image using LZW
        encoded_data = self.encode(diff_bytes)

        # Save compressed file with metadata
        with open(output_file_path, "wb") as f:
            f.write(encoded_data)
            f.write(struct.pack("<I", width))
            f.write(struct.pack("<I", height))
            f.write(struct.pack("B", channels))

        # Calculate statistics
        original_size = os.path.getsize(image_file_path)
        compressed_size = os.path.getsize(output_file_path)
        num_codes = len(diff_bytes)  # Number of symbols encoded
        total_bits = len(encoded_data) * 8
        avg_code_length = utils.LZWUtils.calculate_average_code_length(num_codes, total_bits)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float("inf")

        # Print results
        print(f'Encoded image "{image_file_path}" to "{output_file_path}" successfully.')
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Original image entropy: {original_entropy:.4f} bits/symbol")
        print(f"Difference image entropy: {diff_entropy:.4f} bits/symbol")
        print(f"Average code length: {avg_code_length:.4f} bits/symbol")
        print(f"Compression ratio: {compression_ratio:.4f}")
        print()
