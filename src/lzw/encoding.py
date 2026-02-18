class LZWEncoder:
    # Constructor initializes the encoder with an empty dictionary and sets the initial code size
    def __init__(self) -> None:
        self.dictionary = {bytes([i]): i for i in range(256)}
        self.dictionary_limit = 4096  # Maximum number of entries in the dictionary (12 bits)
        self.reset()

    # Resets the encoder to its initial state
    def reset(self) -> None:
        self.dictionary: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.next_code: int = 256
        self.code_size: int = 9
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

    # Calculates the number of bits needed to represent the current code size
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

    # Create a new file with the encoded data with the .lzw extension
    def encode_file(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as f:
            data = f.read()
        encoded_data = self.encode(data)
        with open(output_file_path, "wb") as f:
            f.write(encoded_data)
