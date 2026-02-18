class LZWDecoder:
    # Initializes the decoder with an empty dictionary and sets the initial code size
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

    # Decodes the chunk of data using the LZW algorithm
    def decode(self, data: bytes) -> bytes:
        self.reset()
        result = bytearray()

        # Load all bytes into the bit buffer
        for byte in data:
            self.bit_buffer |= byte << self.bit_count
            self.bit_count += 8

        # Read first code
        if self.bit_count < self.code_size:
            return b""

        first_code = self.read_code()
        previous_string = self.dictionary[first_code]
        result.extend(previous_string)

        while self.bit_count >= self.code_size:
            code = self.read_code()

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

    # Read a single code from the bit buffer using current code_size
    def read_code(self) -> int:
        code = self.bit_buffer & ((1 << self.code_size) - 1)
        self.bit_buffer >>= self.code_size
        self.bit_count -= self.code_size
        return code

    # Calculates the number of bits needed to represent the current code size
    # Note: Decoder must transition one step earlier than encoder because
    # it adds dictionary entries after reading, while encoder adds before writing
    def calculate_code_size(self) -> int:
        if self.next_code >= (1 << self.code_size) - 1 and self.code_size < 12:
            self.code_size += 1
        return self.code_size

    # Create a new file with the decoded data with the .txt extension
    def decode_file(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as f:
            data = f.read()
        decoded_data = self.decode(data)
        with open(output_file_path, "wb") as f:
            f.write(decoded_data)
