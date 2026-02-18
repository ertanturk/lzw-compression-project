from lzw import decoding, encoding, utils

file_path = "src/samples/sample.txt"
output_file_path = "src/outputs/sample.lzw"

encoding.LZWEncoder().encode_file(file_path, output_file_path)
print(f"Encoded {file_path} to {output_file_path}")
print(utils.LZWUtils.calculate_compression(file_path, output_file_path) + "\n")

decoding.LZWDecoder().decode_file(output_file_path, "src/outputs/sample_decoded.txt")
print(f"Decoded {output_file_path} to src/outputs/sample_decoded.txt")
print(
    utils.LZWUtils.calculate_decompression(output_file_path, "src/outputs/sample_decoded.txt")
    + "\n"
)

if utils.LZWUtils.verify_files(file_path, "src/outputs/sample_decoded.txt"):
    print("Verification successful: Original and decompressed files are identical.")
else:
    print("Verification failed: Original and decompressed files differ.")
