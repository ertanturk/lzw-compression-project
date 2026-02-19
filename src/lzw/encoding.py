import os
import struct

import numpy as np
from PIL import Image

from . import utils


class LZWEncoder:
    # set up the initial dictionary with all single bytes (0-255)
    def __init__(self) -> None:
        self.dictionary = {bytes([i]): i for i in range(256)}
        self.dictionary_limit = 4096  # Maximum number of entries in the dictionary (12 bits)
        self.reset()

    # reset everything so we can encode a new file
    def reset(self) -> None:
        self.dictionary: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.next_code: int = 256
        self.code_size: int = 9  # Initial code size (9 bits to accommodate 256 single-byte entries)
        self.current_string: bytes = b""
        self.bit_buffer: int = 0
        self.bit_count: int = 0

    # main encoding function - takes bytes and returns compressed bytes
    def encode(self, data: bytes) -> bytes:
        self.reset()
        result = bytearray()

        # go through each byte one by one
        for byte in data:
            current_byte = bytes([byte])
            combined_string = self.current_string + current_byte

            if combined_string in self.dictionary:
                self.current_string = combined_string
            else:
                # this combo isnt in the dictionary yet so output current code
                code = self.dictionary[self.current_string]
                result.extend(self.pack_code(code))

                # add new entry to dictionary if theres room
                if self.next_code < self.dictionary_limit:
                    self.dictionary[combined_string] = self.next_code
                    self.next_code += 1
                    self.calculate_code_size()

                self.current_string = current_byte

        # dont forget the last string
        if self.current_string:
            code = self.dictionary[self.current_string]
            result.extend(self.pack_code(code))

        # flush leftover bits
        result.extend(self.flush())
        return bytes(result)

    # bump up code size when dictionary gets too big for current bits
    def calculate_code_size(self) -> int:
        if self.next_code == (1 << self.code_size) and self.code_size < 12:
            self.code_size += 1
        return self.code_size

    # pack a code into the bit buffer and return full bytes when ready
    def pack_code(self, code: int) -> bytes:
        self.bit_buffer |= code << self.bit_count
        self.bit_count += self.code_size
        result = bytearray()
        while self.bit_count >= 8:
            result.append(self.bit_buffer & 0xFF)
            self.bit_buffer >>= 8
            self.bit_count -= 8
        return result

    # push out whatever bits are left in the buffer
    def flush(self) -> bytes:
        result = bytearray()

        while self.bit_count > 0:
            result.append(self.bit_buffer & 0xFF)
            self.bit_buffer >>= 8
            self.bit_count -= 8

        return bytes(result)

    # read a file, compress it with LZW and save the result
    def encode_file(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as f:
            data = f.read()
        encoded_data = self.encode(data)
        with open(output_file_path, "wb") as f:
            f.write(encoded_data)
        print(f'Encoded "{input_file_path}" to "{output_file_path}" successfully.')
        print(f"Calculated code length: {len(encoded_data)} bytes")
        print(utils.LZWUtils.calculate_compression(input_file_path, output_file_path) + "\n")

    # compress an image using difference encoding + LZW
    def encode_image_file(self, image_file_path: str, output_file_path: str) -> None:
        # read the image
        image = Image.open(image_file_path)
        mode = image.mode

        if mode == "L":  # Grayscale image
            # get pixel values as 2D array
            image_array = np.array(image)
            height = len(image_array)
            width = len(image_array[0])
            channels = 1

            # compute differences between neighboring pixels
            diff_image = utils.LZWUtils.compute_difference_image(image_array)

            # turn differences into bytes
            diff_bytes = utils.LZWUtils.difference_to_bytes(diff_image)

        elif mode == "RGB":  # Color image
            image_array = np.array(image)
            height = len(image_array)
            width = len(image_array[0])
            channels = 3

            # do each color channel (R, G, B) separately
            all_diff_bytes = bytearray()

            for channel in range(3):  # 0=Red, 1=Green, 2=Blue
                # grab one channel
                channel_array = image_array[:, :, channel]

                # compute differences for this channel
                diff_image = utils.LZWUtils.compute_difference_image(channel_array)

                # convert to bytes
                channel_bytes = utils.LZWUtils.difference_to_bytes(diff_image)
                all_diff_bytes.extend(channel_bytes)

            diff_bytes = bytes(all_diff_bytes)
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

        # entropy of original image
        original_bytes, _, _, _ = utils.LZWUtils.open_image_file(image_file_path)
        original_entropy = utils.LZWUtils.calculate_entropy(original_bytes)

        # entropy of the difference image
        diff_entropy = utils.LZWUtils.calculate_entropy(diff_bytes)

        # compress with LZW
        encoded_data = self.encode(diff_bytes)

        # save compressed file with dimensions at the end
        with open(output_file_path, "wb") as f:
            f.write(encoded_data)
            f.write(struct.pack("<I", width))  # 4 bytes for width
            f.write(struct.pack("<I", height))  # 4 bytes for height
            f.write(struct.pack("B", channels))  # 1 byte for channels

        # calculate and print stats
        original_size = os.path.getsize(image_file_path)
        compressed_size = os.path.getsize(output_file_path)

        # how many symbols we encoded
        num_symbols = len(diff_bytes)

        # total bits in output
        total_bits = len(encoded_data) * 8

        # average bits per symbol
        avg_code_length = utils.LZWUtils.calculate_average_code_length(num_symbols, total_bits)

        # compression ratio
        if compressed_size > 0:
            compression_ratio = original_size / compressed_size
        else:
            compression_ratio = 0

        # print everything
        print(f'Encoded image "{image_file_path}" to "{output_file_path}" successfully.')
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Original image entropy: {original_entropy:.4f} bits/symbol")
        print(f"Difference image entropy: {diff_entropy:.4f} bits/symbol")
        print(f"Average code length: {avg_code_length:.4f} bits/symbol")
        print(f"Compression ratio: {compression_ratio:.4f}")
        print()
