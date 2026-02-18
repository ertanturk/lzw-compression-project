from lzw import decoding, encoding, utils

file_path = "src/samples/sample.txt"
output_file_path = "src/outputs/sample.lzw"
grayscale_image_file_path = "src/samples/big_image_grayscale.bmp"
rgb_image_file_path = "src/samples/big_image.bmp"

encoding.LZWEncoder().encode_file(file_path, output_file_path)
decoding.LZWDecoder().decode_file(output_file_path, "src/outputs/sample_decoded.txt")
utils.LZWUtils.verify_files(file_path, "src/outputs/sample_decoded.txt")

encoding.LZWEncoder().encode_image_file(
    grayscale_image_file_path, "src/outputs/grayscale_image.lzw"
)
decoding.LZWDecoder().decode_image_file(
    "src/outputs/grayscale_image.lzw", "src/outputs/grayscale_image_decoded.bmp"
)
utils.LZWUtils.verify_image_files(
    grayscale_image_file_path, "src/outputs/grayscale_image_decoded.bmp"
)

encoding.LZWEncoder().encode_image_file(rgb_image_file_path, "src/outputs/rgb_image.lzw")
decoding.LZWDecoder().decode_image_file(
    "src/outputs/rgb_image.lzw", "src/outputs/rgb_image_decoded.bmp"
)
utils.LZWUtils.verify_image_files(rgb_image_file_path, "src/outputs/rgb_image_decoded.bmp")
