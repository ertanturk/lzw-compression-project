import numpy as np
from PIL import Image

from lzw import decoding, encoding, utils

txt_file_path = "src/samples/sample.txt"
img_file_path = "src/samples/big_image.bmp"

txt_file_output_path = "src/outputs/sample_compressed.lzw"
img_file_output_path = "src/outputs/big_image_compressed.lzw"

lvl1_check = False
lvl2_check = False
lvl3_check = False
lvl4_check = False
lvl5_check = False
total_passed = 0
# Level 1: LZW Encoding and Decoding (Text)
try:
    # encode the text file
    encoder = encoding.LZWEncoder()
    encoder.encode_file(txt_file_path, txt_file_output_path)

    # decode the text file
    decoder = decoding.LZWDecoder()
    decoder.decode_file(txt_file_output_path, "src/outputs/sample_decompressed.txt")

    # verify that the decompressed text matches the original
    with open(txt_file_path, encoding="utf-8") as f:
        original_text = f.read()
    with open("src/outputs/sample_decompressed.txt", encoding="utf-8") as f:
        decompressed_text = f.read()
    if original_text != decompressed_text:
        raise ValueError("Decompressed text does not match original")

except Exception as e:
    print(f"Error during text file encoding/decoding: {e}")
else:
    lvl1_check = True
    total_passed += 1
    print("Text file encoding and decoding completed successfully.\n")

# Level 2: Image Compression (Gray Level)
try:
    # encode the image file
    encoder = encoding.LZWEncoder()
    encoder.encode_image_file(img_file_path, img_file_output_path)

    # decode the image file
    decoder = decoding.LZWDecoder()
    decoder.decode_image_file(img_file_output_path, "src/outputs/big_image_decompressed.bmp")

    # verify that the decompressed image matches the original
    original_image = Image.open(img_file_path)
    decompressed_image = Image.open("src/outputs/big_image_decompressed.bmp")
    if not np.array_equal(np.array(original_image), np.array(decompressed_image)):
        raise ValueError("Decompressed image does not match original")
except Exception as e:
    print(f"Error during image file encoding/decoding: {e}")
else:
    lvl2_check = True
    total_passed += 1
    print("Image file encoding and decoding completed successfully.\n")

# Level 3: Image Compression (Gray Level differences)
try:
    # convert the color image to grayscale first
    gray_image = Image.open(img_file_path).convert("L")
    gray_image.save("src/outputs/big_image_grayscale.bmp")
    gray_array = np.array(gray_image)
    height, width = gray_array.shape

    # compute difference image (row-wise and column-wise)
    diff_image = utils.LZWUtils.compute_difference_image(gray_array)

    # convert differences to bytes
    diff_bytes = utils.LZWUtils.difference_to_bytes(diff_image)

    # calculate entropy before and after differencing
    original_gray_bytes = gray_array.flatten().tobytes()
    original_entropy = utils.LZWUtils.calculate_entropy(original_gray_bytes)
    diff_entropy = utils.LZWUtils.calculate_entropy(diff_bytes)
    print(f"Grayscale original entropy: {original_entropy:.4f} bits/symbol")
    print(f"Grayscale difference entropy: {diff_entropy:.4f} bits/symbol")

    # encode the difference bytes with LZW
    encoder = encoding.LZWEncoder()
    encoded_diff = encoder.encode(diff_bytes)
    print(f"Compressed size: {len(encoded_diff)} bytes")

    # decode it back
    decoder = decoding.LZWDecoder()
    decoded_diff = decoder.decode(encoded_diff)

    # restore difference image from bytes
    restored_diff = utils.LZWUtils.bytes_to_difference(decoded_diff, height, width)

    # restore original pixel values from differences
    restored_pixels = utils.LZWUtils.restore_from_difference(restored_diff)
    restored_array = np.array(restored_pixels, dtype=np.uint8)

    # save the restored image and compare
    restored_image = Image.fromarray(restored_array, mode="L")
    restored_image.save("src/outputs/big_image_grayscale_diff_restored.bmp")

    # check if they match
    if np.array_equal(gray_array, restored_array):
        print("Verification: grayscale difference round-trip matches!\n")
    else:
        print("Verification FAILED: images dont match\n")
        raise ValueError("Restored grayscale image does not match original")

except Exception as e:
    print(f"Error during grayscale difference encoding/decoding: {e}")
else:
    lvl3_check = True
    total_passed += 1
    print("Grayscale difference encoding and decoding completed successfully.\n")

# Level 4: Image Compression (Color)
try:
    # apply color isolation (red, green, blue) to the image and save the modified image
    image_array = np.array(Image.open(img_file_path))
    modified_image_array_red = Image.fromarray(
        utils.LZWUtils.isolate_color_channel(image_array, channel="red")
    )  # Red
    modified_image_array_green = Image.fromarray(
        utils.LZWUtils.isolate_color_channel(image_array, channel="green")
    )  # Green
    modified_image_array_blue = Image.fromarray(
        utils.LZWUtils.isolate_color_channel(image_array, channel="blue")
    )  # Blue

    # save the modified images
    modified_image_array_red.save("src/outputs/big_image_red_channel.png")
    modified_image_array_green.save("src/outputs/big_image_green_channel.png")
    modified_image_array_blue.save("src/outputs/big_image_blue_channel.png")

    # Apply LZW coding to each color components separately and save the compressed files
    encoder = encoding.LZWEncoder()
    encoder.encode_image_file(
        "src/outputs/big_image_red_channel.png", "src/outputs/big_image_red_channel_compressed.lzw"
    )
    encoder.encode_image_file(
        "src/outputs/big_image_green_channel.png",
        "src/outputs/big_image_green_channel_compressed.lzw",
    )
    encoder.encode_image_file(
        "src/outputs/big_image_blue_channel.png",
        "src/outputs/big_image_blue_channel_compressed.lzw",
    )

    # decode the compressed color channel files and save the restored images
    decoder = decoding.LZWDecoder()
    decoder.decode_image_file(
        "src/outputs/big_image_red_channel_compressed.lzw",
        "src/outputs/big_image_red_channel_decompressed.png",
    )
    decoder.decode_image_file(
        "src/outputs/big_image_green_channel_compressed.lzw",
        "src/outputs/big_image_green_channel_decompressed.png",
    )
    decoder.decode_image_file(
        "src/outputs/big_image_blue_channel_compressed.lzw",
        "src/outputs/big_image_blue_channel_decompressed.png",
    )

    # verify that the restored images match the original color channels
    original_image = Image.open(img_file_path)
    original_array = np.array(original_image)
    red_channel = utils.LZWUtils.isolate_color_channel(original_array, channel="red")
    green_channel = utils.LZWUtils.isolate_color_channel(original_array, channel="green")
    blue_channel = utils.LZWUtils.isolate_color_channel(original_array, channel="blue")
    restored_red = np.array(Image.open("src/outputs/big_image_red_channel_decompressed.png"))
    restored_green = np.array(Image.open("src/outputs/big_image_green_channel_decompressed.png"))
    restored_blue = np.array(Image.open("src/outputs/big_image_blue_channel_decompressed.png"))

    if not np.array_equal(red_channel, restored_red):
        raise ValueError("Red channel does not match")
    if not np.array_equal(green_channel, restored_green):
        raise ValueError("Green channel does not match")
    if not np.array_equal(blue_channel, restored_blue):
        raise ValueError("Blue channel does not match")

except Exception as e:
    print(f"Error during color image file encoding/decoding: {e}")
else:
    lvl4_check = True
    total_passed += 1
    print("Color image file encoding and decoding completed successfully.\n")

# Level 5: Image Compression (Color differences)
try:
    # open the RGB image
    color_image = Image.open(img_file_path)
    color_array = np.array(color_image)
    height, width, _ = color_array.shape

    channel_names = ["red", "green", "blue"]
    all_encoded = []
    all_channel_arrays = []

    for ch in range(3):
        # grab one channel
        channel_array = color_array[:, :, ch]
        all_channel_arrays.append(channel_array)

        # compute differences for this channel
        diff_image = utils.LZWUtils.compute_difference_image(channel_array)
        diff_bytes = utils.LZWUtils.difference_to_bytes(diff_image)

        # entropy stats
        ch_entropy = utils.LZWUtils.calculate_entropy(channel_array.flatten().tobytes())
        diff_entropy = utils.LZWUtils.calculate_entropy(diff_bytes)
        print(
            f"{channel_names[ch]} channel - original entropy: {ch_entropy:.4f}, "
            f"difference entropy: {diff_entropy:.4f}"
        )

        # encode with LZW
        encoder = encoding.LZWEncoder()
        encoded_data = encoder.encode(diff_bytes)
        all_encoded.append(encoded_data)
        print(f"{channel_names[ch]} channel - compressed size: {len(encoded_data)} bytes")

    # decode each channel and restore from differences
    restored_color = np.zeros((height, width, 3), dtype=np.uint8)

    for ch in range(3):
        # decode
        decoder = decoding.LZWDecoder()
        decoded_data = decoder.decode(all_encoded[ch])

        # bytes -> difference image -> original pixels
        diff_image = utils.LZWUtils.bytes_to_difference(decoded_data, height, width)
        restored_pixels = utils.LZWUtils.restore_from_difference(diff_image)

        # put channel back into the RGB array
        for row in range(height):
            for col in range(width):
                restored_color[row][col][ch] = restored_pixels[row][col]

    # save restored image
    restored_image = Image.fromarray(restored_color, mode="RGB")
    restored_image.save("src/outputs/big_image_color_diff_restored.bmp")

    # verify
    if np.array_equal(color_array, restored_color):
        print("Verification: color difference round-trip matches!\n")
    else:
        print("Verification FAILED: images dont match\n")
        raise ValueError("Restored color image does not match original")

except Exception as e:
    print(f"Error during color difference encoding/decoding: {e}")
else:
    lvl5_check = True
    total_passed += 1
    print("Color difference encoding and decoding completed successfully.\n")


print("Summary of checks:")
print(f"Level 1 (Text LZW): {'PASS' if lvl1_check else 'FAIL'}")
print(f"Level 2 (Image LZW): {'PASS' if lvl2_check else 'FAIL'}")
print(f"Level 3 (Grayscale differences): {'PASS' if lvl3_check else 'FAIL'}")
print(f"Level 4 (Color LZW): {'PASS' if lvl4_check else 'FAIL'}")
print(f"Level 5 (Color differences): {'PASS' if lvl5_check else 'FAIL'}")
print(f"Total passed: {total_passed}/5")
