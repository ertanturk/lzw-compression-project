# LZW Compression Tool

A tool for compressing and decompressing text and image files using the LZW algorithm. It has a GUI built with tkinter.

## What it does

- Compress text files (.txt) and image files (.png, .bmp)
- Decompress .lzw files back to original
- Two methods for images: gray levels and differences
- Color mode options: default, red, green, blue, grayscale
- Shows compression stats like entropy, compression ratio, compression factor and space savings
- Preview files before and after compression

## How to install

You need Python 3.12 or newer.

```bash
git clone https://github.com/username/lzw-compression-project.git
cd lzw-compression-project
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## How to run

```bash
python src/gui/app.py
```

This opens a window with two tabs: Compress and Decompress.

### Compress

1. Click "Browse" and pick a file
2. If its an image you can change color mode and method
3. Click "Compress"
4. Check the stats and click "Save as .lzw File"

### Decompress

1. Click "Browse" and pick a .lzw file
2. Click "Decompress"
3. Preview the result and click "Save Decompressed File"

## Project structure

```
src/
  lzw/
    encoding.py    - LZW encoder
    decoding.py    - LZW decoder
    utils.py       - helper functions and GUI utilities
  gui/
    app.py         - the GUI application
  samples/         - sample files for testing
  outputs/         - compressed/decompressed output files
  tests/
    test.py        - test script
```

## Dependencies

- Pillow (for images)
- numpy (for array stuff)
