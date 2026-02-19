import struct

import numpy as np
from PIL import Image

from . import utils


class LZWDecoder:
    # set up the initial dictionary with all single bytes (0-255)
    def __init__(self) -> None:
        self.dictionary = {i: bytes([i]) for i in range(256)}
        self.dictionary_limit = 4096  # Maximum number of entries in the dictionary (12 bits)
        self.reset()

    # reset everything so we can decode a new file
    def reset(self) -> None:
        self.dictionary = {i: bytes([i]) for i in range(256)}
        self.next_code: int = 256
        self.code_size: int = 9
        self.bit_buffer: int = 0
        self.bit_count: int = 0
        self.data: bytes = b""
        self.data_index: int = 0

    # main decoding function - takes compressed bytes and returns original bytes
    def decode(self, data: bytes) -> bytes:
        self.reset()
        self.data = data
        result = bytearray()

        # read first code
        first_code = self.read_code()
        if first_code is None:
            return b""

        previous_string = self.dictionary[first_code]
        result.extend(previous_string)

        while True:
            code = self.read_code()
            if code is None:
                break

            if code in self.dictionary:
                current_string = self.dictionary[code]
            elif code == self.next_code:
                current_string = previous_string + previous_string[:1]
            else:
                raise ValueError(f"Invalid LZW code: {code}")

            result.extend(current_string)

            # add new dictionary entry if theres room
            if self.next_code < self.dictionary_limit:
                self.dictionary[self.next_code] = previous_string + current_string[:1]
                self.next_code += 1
                self.calculate_code_size()

            previous_string = current_string

        return bytes(result)

    # read one code from the bit buffer
    def read_code(self) -> int | None:
        # load more bytes if we need them
        while self.bit_count < self.code_size and self.data_index < len(self.data):
            self.bit_buffer |= self.data[self.data_index] << self.bit_count
            self.bit_count += 8
            self.data_index += 1

        if self.bit_count < self.code_size:
            return None

        code = self.bit_buffer & ((1 << self.code_size) - 1)
        self.bit_buffer >>= self.code_size
        self.bit_count -= self.code_size
        return code

    # bump up code size when dictionary gets big enough
    # decoder needs to do this one step earlier than encoder
    # because it adds entries after reading, encoder adds before writing
    def calculate_code_size(self) -> int:
        if self.next_code >= (1 << self.code_size) - 1 and self.code_size < 12:
            self.code_size += 1
        return self.code_size

    # read a compressed file, decode it and save the result
    def decode_file(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as f:
            data = f.read()
        decoded_data = self.decode(data)
        with open(output_file_path, "wb") as f:
            f.write(decoded_data)
        print(f'Decoded "{input_file_path}" to "{output_file_path}" successfully.')
        print(f"Calculated code length: {len(decoded_data)} bytes" + "\n")

    # decompress an image file and restore from differences
    def decode_image_file(self, input_file_path: str, output_file_path: str) -> None:
        # read the compressed file
        with open(input_file_path, "rb") as f:
            # metadata is at the end of the file (9 bytes: 4+4+1)
            f.seek(-9, 2)  # go 9 bytes before end
            width = struct.unpack("<I", f.read(4))[0]
            height = struct.unpack("<I", f.read(4))[0]
            channels = struct.unpack("B", f.read(1))[0]

            # read the actual compressed data (everything except last 9 bytes)
            f.seek(0, 2)  # go to end
            file_size = f.tell()
            f.seek(0)  # go back to start
            compressed_data = f.read(file_size - 9)

        # decode LZW to get the difference image bytes
        decoded_data = self.decode(compressed_data)

        # restore the original image
        if channels == 1:  # Grayscale
            # bytes -> difference image -> original image
            diff_image = utils.LZWUtils.bytes_to_difference(decoded_data, height, width)
            restored = utils.LZWUtils.restore_from_difference(diff_image)
            restored_array = np.array(restored, dtype=np.uint8)
            image = Image.fromarray(restored_array, mode="L")

        elif channels == 3:  # RGB
            # each channel has height * width * 2 bytes
            bytes_per_channel = height * width * 2

            # restore each color channel separately
            restored_channels: list = []  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]

            for channel in range(3):  # 0=Red, 1=Green, 2=Blue
                # get this channel's bytes
                start = channel * bytes_per_channel
                end = start + bytes_per_channel
                channel_bytes = decoded_data[start:end]

                # bytes -> differences -> original channel
                diff_image = utils.LZWUtils.bytes_to_difference(channel_bytes, height, width)
                restored_channel = utils.LZWUtils.restore_from_difference(diff_image)
                restored_channels.append(restored_channel)  # type: ignore

            # put R, G, B channels back together
            restored_array = np.zeros((height, width, 3), dtype=np.uint8)
            for row in range(height):
                for col in range(width):
                    restored_array[row][col][0] = restored_channels[0][row][col]  # Red
                    restored_array[row][col][1] = restored_channels[1][row][col]  # Green
                    restored_array[row][col][2] = restored_channels[2][row][col]  # Blue

            image = Image.fromarray(restored_array, mode="RGB")
        else:
            raise ValueError(f"Unsupported channel count: {channels}")

        # save it
        image.save(output_file_path)
        print(f'Decoded image "{input_file_path}" to "{output_file_path}" successfully.')
        print(f"Restored image size: {width}x{height}, channels: {channels}\n")
