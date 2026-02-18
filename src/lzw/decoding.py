import struct

import numpy as np
from PIL import Image

from . import utils


class LZWDecoder:
    # Initializes the decoder with the
    # base dictionary (256 single-byte entries) and sets the initial code size
    def __init__(self) -> None:
        self.dictionary = {i: bytes([i]) for i in range(256)}
        self.dictionary_limit = 4096  # Maximum number of entries in the dictionary (12 bits)
        self.reset()

    # Resets the decoder to its initial state
    def reset(self) -> None:
        self.dictionary = {i: bytes([i]) for i in range(256)}
        self.next_code: int = 256
        self.code_size: int = 9
        self.bit_buffer: int = 0
        self.bit_count: int = 0
        self.data: bytes = b""
        self.data_index: int = 0

    # Decodes the chunk of data using the LZW algorithm
    def decode(self, data: bytes) -> bytes:
        self.reset()
        self.data = data
        result = bytearray()

        # Read first code
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

            # Add new entry to the dictionary if there's room
            if self.next_code < self.dictionary_limit:
                self.dictionary[self.next_code] = previous_string + current_string[:1]
                self.next_code += 1
                self.calculate_code_size()

            previous_string = current_string

        return bytes(result)

    # Read a single code from the bit buffer, loading bytes as needed
    def read_code(self) -> int | None:
        # Load more bytes into the buffer as needed
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

    # Increases the code size when the dictionary grows beyond the current bit width limit
    # Note: Decoder must transition one step earlier than encoder because
    # it adds dictionary entries after reading, while encoder adds before writing
    def calculate_code_size(self) -> int:
        if self.next_code >= (1 << self.code_size) - 1 and self.code_size < 12:
            self.code_size += 1
        return self.code_size

    # Reads an LZW-encoded file, decodes its contents, and writes the result to the output file
    def decode_file(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as f:
            data = f.read()
        decoded_data = self.decode(data)
        with open(output_file_path, "wb") as f:
            f.write(decoded_data)
        print(f'Decoded "{input_file_path}" to "{output_file_path}" successfully.')
        print(f"Calculated code length: {len(decoded_data)} bytes" + "\n")

    # Reads an LZW-encoded image file, restores the difference image,
    # reconstructs the original image, and saves it
    def decode_image_file(self, input_file_path: str, output_file_path: str) -> None:
        # Step 1: Read the compressed file
        with open(input_file_path, "rb") as f:
            # Read metadata from the end of the file (9 bytes: 4 + 4 + 1)
            f.seek(-9, 2)  # Go to 9 bytes before end of file
            width = struct.unpack("<I", f.read(4))[0]  # Read width (4 bytes)
            height = struct.unpack("<I", f.read(4))[0]  # Read height (4 bytes)
            channels = struct.unpack("B", f.read(1))[0]  # Read channels (1 byte)

            # Read the compressed data (everything except the last 9 bytes)
            f.seek(0, 2)  # Go to end of file
            file_size = f.tell()  # Get file size
            f.seek(0)  # Go back to start
            compressed_data = f.read(file_size - 9)  # Read all but last 9 bytes

        # Step 2: Decode LZW to get difference image bytes
        decoded_data = self.decode(compressed_data)

        # Step 3: Restore original image from difference image
        if channels == 1:  # Grayscale image
            # Convert bytes back to difference image (2D list)
            diff_image = utils.LZWUtils.bytes_to_difference(decoded_data, height, width)

            # Restore original image from differences
            restored = utils.LZWUtils.restore_from_difference(diff_image)

            # Convert 2D list to numpy array for PIL
            restored_array = np.array(restored, dtype=np.uint8)
            image = Image.fromarray(restored_array, mode="L")

        elif channels == 3:  # RGB image
            # Each channel has height * width * 2 bytes (2 bytes per pixel)
            bytes_per_channel = height * width * 2

            # Restore each color channel separately
            restored_channels: list = []  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]

            for channel in range(3):  # 0=Red, 1=Green, 2=Blue
                # Extract bytes for this channel
                start = channel * bytes_per_channel
                end = start + bytes_per_channel
                channel_bytes = decoded_data[start:end]

                # Convert bytes to difference image
                diff_image = utils.LZWUtils.bytes_to_difference(channel_bytes, height, width)

                # Restore original channel from differences
                restored_channel = utils.LZWUtils.restore_from_difference(diff_image)
                restored_channels.append(restored_channel)  # type: ignore

            # Combine RGB channels into one image
            # Create 3D array: height x width x 3
            restored_array = np.zeros((height, width, 3), dtype=np.uint8)
            for row in range(height):
                for col in range(width):
                    restored_array[row][col][0] = restored_channels[0][row][col]  # Red
                    restored_array[row][col][1] = restored_channels[1][row][col]  # Green
                    restored_array[row][col][2] = restored_channels[2][row][col]  # Blue

            image = Image.fromarray(restored_array, mode="RGB")
        else:
            raise ValueError(f"Unsupported channel count: {channels}")

        # Step 4: Save the restored image
        image.save(output_file_path)
        print(f'Decoded image "{input_file_path}" to "{output_file_path}" successfully.')
        print(f"Restored image size: {width}x{height}, channels: {channels}\n")
