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
        # Step 1: Read the image file
        image = Image.open(image_file_path)
        mode = image.mode

        if mode == "L":  # Grayscale image
            # Convert image to 2D list of pixel values
            image_array = np.array(image)
            height = len(image_array)
            width = len(image_array[0])
            channels = 1

            # Step 2: Compute difference image
            diff_image = utils.LZWUtils.compute_difference_image(image_array)

            # Step 3: Convert difference image to bytes
            diff_bytes = utils.LZWUtils.difference_to_bytes(diff_image)

        elif mode == "RGB":  # Color image
            image_array = np.array(image)
            height = len(image_array)
            width = len(image_array[0])
            channels = 3

            # Process each color channel (Red, Green, Blue) separately
            all_diff_bytes = bytearray()

            for channel in range(3):  # 0=Red, 1=Green, 2=Blue
                # Extract one channel as 2D array
                channel_array = image_array[:, :, channel]

                # Compute difference image for this channel
                diff_image = utils.LZWUtils.compute_difference_image(channel_array)

                # Convert to bytes and add to result
                channel_bytes = utils.LZWUtils.difference_to_bytes(diff_image)
                all_diff_bytes.extend(channel_bytes)

            diff_bytes = bytes(all_diff_bytes)
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

        # Step 4: Calculate entropy of original image (for comparison)
        original_bytes, _, _, _ = utils.LZWUtils.open_image_file(image_file_path)
        original_entropy = utils.LZWUtils.calculate_entropy(original_bytes)

        # Step 5: Calculate entropy of difference image
        diff_entropy = utils.LZWUtils.calculate_entropy(diff_bytes)

        # Step 6: Encode difference image using LZW algorithm
        encoded_data = self.encode(diff_bytes)

        # Step 7: Save compressed file with metadata at the end
        with open(output_file_path, "wb") as f:
            # Write compressed data
            f.write(encoded_data)
            # Write image dimensions (needed for decoding)
            f.write(struct.pack("<I", width))  # 4 bytes for width
            f.write(struct.pack("<I", height))  # 4 bytes for height
            f.write(struct.pack("B", channels))  # 1 byte for channels

        # Step 8: Calculate and print statistics
        original_size = os.path.getsize(image_file_path)
        compressed_size = os.path.getsize(output_file_path)

        # Number of symbols we encoded
        num_symbols = len(diff_bytes)

        # Total bits in compressed output
        total_bits = len(encoded_data) * 8

        # Average bits per symbol
        avg_code_length = utils.LZWUtils.calculate_average_code_length(num_symbols, total_bits)

        # Compression ratio = original size / compressed size
        if compressed_size > 0:
            compression_ratio = original_size / compressed_size
        else:
            compression_ratio = 0

        # Print results
        print(f'Encoded image "{image_file_path}" to "{output_file_path}" successfully.')
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Original image entropy: {original_entropy:.4f} bits/symbol")
        print(f"Difference image entropy: {diff_entropy:.4f} bits/symbol")
        print(f"Average code length: {avg_code_length:.4f} bits/symbol")
        print(f"Compression ratio: {compression_ratio:.4f}")
        print()
