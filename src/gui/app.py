# LZW compression tool gui

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

# need this so python can find the lzw package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lzw.utils import GUIUtils


class LZWApp:
    def __init__(self) -> None:
        # main window setup
        self.root = tk.Tk()
        self.root.title("LZW Compression Tool")
        self.root.geometry("900x750")
        self.root.minsize(700, 600)

        # -- compress tab stuff --
        self.compress_file_path: str = ""
        self.compress_file_type: str = ""
        self.compressed_bytes: bytes = b""
        self.preview_photo: ImageTk.PhotoImage | None = None

        # -- decompress tab stuff --
        self.decompress_file_path: str = ""
        self.decompressed_data: bytes | Image.Image | None = None
        self.decompressed_type: str = ""
        self.decompressed_ext: str = ""
        self.decompress_photo: ImageTk.PhotoImage | None = None

        # tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create the Compress tab
        self.compress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.compress_tab, text="  Compress  ")

        self.decompress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.decompress_tab, text="  Decompress  ")

        # build each tab
        self.build_compress_tab()
        self.build_decompress_tab()

    # =========================================================================
    # COMPRESS TAB
    # =========================================================================

    # set up all the compress tab widgets
    def build_compress_tab(self) -> None:
        # -- file selection --
        file_frame = ttk.LabelFrame(self.compress_tab, text="Step 1: Select a File")
        file_frame.pack(fill="x", padx=10, pady=5)

        select_btn = ttk.Button(file_frame, text="Browse...", command=self.select_compress_file)
        select_btn.pack(side="left", padx=5, pady=5)

        self.compress_file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.compress_file_label.pack(side="left", padx=10, pady=5)

        # -- preview area --
        preview_frame = ttk.LabelFrame(self.compress_tab, text="Step 2: Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # text preview
        self.compress_text_preview = scrolledtext.ScrolledText(
            preview_frame, wrap="word", height=10, state="disabled"
        )

        # image preview
        self.compress_image_label = tk.Label(preview_frame, bg="white")

        # shown when nothing is loaded yet
        self.compress_preview_msg = ttk.Label(
            preview_frame, text="No file loaded", foreground="gray", anchor="center"
        )
        self.compress_preview_msg.pack(fill="both", expand=True)

        # -- image options --
        options_frame = ttk.LabelFrame(self.compress_tab, text="Step 3: Options (Images Only)")
        options_frame.pack(fill="x", padx=10, pady=5)

        # color mode picker
        ttk.Label(options_frame, text="Color Mode:").pack(side="left", padx=(10, 5), pady=5)
        self.color_var = tk.StringVar(value="default")
        self.color_combo = ttk.Combobox(
            options_frame,
            textvariable=self.color_var,
            values=["default", "red", "green", "blue", "grayscale"],
            state="disabled",
            width=12,
        )
        self.color_combo.pack(side="left", padx=5, pady=5)
        # update preview when color mode changes
        self.color_combo.bind("<<ComboboxSelected>>", self.on_color_mode_changed)

        # method picker
        ttk.Label(options_frame, text="Method:").pack(side="left", padx=(20, 5), pady=5)
        self.method_var = tk.StringVar(value="differences")
        self.method_combo = ttk.Combobox(
            options_frame,
            textvariable=self.method_var,
            values=["gray_levels", "differences"],
            state="disabled",
            width=12,
        )
        self.method_combo.pack(side="left", padx=5, pady=5)

        # -- compress button --
        self.compress_btn = ttk.Button(
            self.compress_tab, text="Compress", command=self.do_compress, state="disabled"
        )
        self.compress_btn.pack(pady=5)

        # -- stats section --
        stats_frame = ttk.LabelFrame(self.compress_tab, text="Compression Statistics")
        stats_frame.pack(fill="x", padx=10, pady=5)

        self.stats_label = ttk.Label(
            stats_frame, text="No compression performed yet", foreground="gray"
        )
        self.stats_label.pack(padx=10, pady=5, anchor="w")

        # -- save button --
        self.save_compress_btn = ttk.Button(
            self.compress_tab,
            text="Save as .lzw File",
            command=self.save_compressed_file,
            state="disabled",
        )
        self.save_compress_btn.pack(pady=5)

    # let user pick a file to compress
    def select_compress_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a file to compress",
            filetypes=[
                ("All Supported", "*.txt *.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("Text Files", "*.txt"),
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("All Files", "*.*"),
            ],
        )
        # if user cancelled just skip
        if not file_path:
            return

        # save file info
        self.compress_file_path = file_path
        self.compress_file_type = GUIUtils.get_file_type(file_path)
        self.compressed_bytes = b""  # clear old compression

        # show file name
        file_name = os.path.basename(file_path)
        self.compress_file_label.configure(text=file_name, foreground="black")

        # enable compress button
        self.compress_btn.configure(state="normal")

        # reset stats and save
        self.stats_label.configure(text="No compression performed yet", foreground="gray")
        self.save_compress_btn.configure(state="disabled")

        # turn on image options if its an image
        if self.compress_file_type == "image":
            self.color_combo.configure(state="readonly")
            self.method_combo.configure(state="readonly")
            self.color_var.set("default")
            self.method_var.set("differences")
        else:
            self.color_combo.configure(state="disabled")
            self.method_combo.configure(state="disabled")
            self.color_var.set("default")
            self.method_var.set("differences")

        # show preview of the file
        self.show_compress_preview()

    # show a preview of whatever file was selected
    def show_compress_preview(self) -> None:
        # hide everything first
        self.compress_text_preview.pack_forget()
        self.compress_image_label.pack_forget()
        self.compress_preview_msg.pack_forget()

        if self.compress_file_type == "text":
            # text file preview
            try:
                with open(self.compress_file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read(10000)  # just the first 10k chars
                self.compress_text_preview.configure(state="normal")
                self.compress_text_preview.delete("1.0", "end")
                self.compress_text_preview.insert("1.0", content)
                self.compress_text_preview.configure(state="disabled")
                self.compress_text_preview.pack(fill="both", expand=True, padx=5, pady=5)
            except Exception as e:
                self.compress_preview_msg.configure(text=f"Cannot preview: {e}")
                self.compress_preview_msg.pack(fill="both", expand=True)

        elif self.compress_file_type == "image":
            # image preview
            self.update_image_preview()

    # refresh the image preview when color mode changes
    def update_image_preview(self, event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        # hide other stuff
        self.compress_text_preview.pack_forget()
        self.compress_preview_msg.pack_forget()

        try:
            # get color mode and apply it
            color_mode = self.color_var.get()
            image = GUIUtils.apply_color_mode(self.compress_file_path, color_mode)

            # shrink it to fit the preview area
            image_copy = image.copy()
            image_copy.thumbnail((400, 300))

            # tkinter needs PhotoImage to display
            self.preview_photo = ImageTk.PhotoImage(image_copy)
            self.compress_image_label.configure(image=self.preview_photo)
            self.compress_image_label.pack(fill="both", expand=True, padx=5, pady=5)
        except Exception as e:
            self.compress_image_label.pack_forget()
            self.compress_preview_msg.configure(text=f"Cannot preview image: {e}")
            self.compress_preview_msg.pack(fill="both", expand=True)

    # update preview when color dropdown changes
    def on_color_mode_changed(self, event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        if self.compress_file_type == "image" and self.compress_file_path:
            self.update_image_preview()

    # run the actual compression
    def do_compress(self) -> None:
        if not self.compress_file_path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        try:
            if self.compress_file_type == "text":
                # text compression
                lzw_bytes, stats = GUIUtils.compress_text(self.compress_file_path)

                # show stats
                stats_text = (
                    f"Original Size: {stats['original_size']} bytes\n"
                    f"Compressed Size: {stats['compressed_size']} bytes\n"
                    f"Entropy: {stats['entropy']:.4f} bits/symbol\n"
                    f"Average Code Length: {stats['avg_code_length']:.4f} bits/symbol\n"
                    f"Compression Ratio (CR): {stats['compression_ratio']:.4f}\n"
                    f"Compression Factor (CF): {stats['compression_factor']:.4f}\n"
                    f"Space Savings (SS): {stats['space_savings']:.2%}"
                )

            elif self.compress_file_type == "image":
                # image compression
                color_mode = self.color_var.get()
                method = self.method_var.get()
                lzw_bytes, stats = GUIUtils.compress_image(
                    self.compress_file_path, color_mode, method
                )

                # show stats
                stats_text = (
                    f"Original Size: {stats['original_size']} bytes\n"
                    f"Compressed Size: {stats['compressed_size']} bytes\n"
                    f"Original Entropy: {stats['original_entropy']:.4f} bits/symbol\n"
                    f"Data Entropy: {stats['data_entropy']:.4f} bits/symbol\n"
                    f"Average Code Length: {stats['avg_code_length']:.4f} bits/symbol\n"
                    f"Compression Ratio (CR): {stats['compression_ratio']:.4f}\n"
                    f"Compression Factor (CF): {stats['compression_factor']:.4f}\n"
                    f"Space Savings (SS): {stats['space_savings']:.2%}"
                )
            else:
                messagebox.showerror("Error", "Unknown file type.")
                return

            # save result and update buttons
            self.compressed_bytes = lzw_bytes
            self.stats_label.configure(text=stats_text, foreground="black")
            self.save_compress_btn.configure(state="normal")
            messagebox.showinfo("Success", "Compression completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Compression failed:\n{e}")

    # save compressed data to .lzw file
    def save_compressed_file(self) -> None:
        if not self.compressed_bytes:
            messagebox.showwarning("Warning", "No compressed data to save.")
            return

        # suggest a name based on original file
        original_name = os.path.basename(self.compress_file_path)
        suggested_name = os.path.splitext(original_name)[0] + ".lzw"

        file_path = filedialog.asksaveasfilename(
            title="Save compressed file",
            defaultextension=".lzw",
            initialfile=suggested_name,
            filetypes=[("LZW Files", "*.lzw"), ("All Files", "*.*")],
        )
        # if cancelled do nothing
        if not file_path:
            return

        try:
            GUIUtils.save_lzw_file(self.compressed_bytes, file_path)
            messagebox.showinfo("Saved", f"File saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    # =========================================================================
    # DECOMPRESS TAB
    # =========================================================================

    # set up all the decompress tab widgets
    def build_decompress_tab(self) -> None:
        # -- file selection --
        file_frame = ttk.LabelFrame(self.decompress_tab, text="Step 1: Select .lzw File")
        file_frame.pack(fill="x", padx=10, pady=5)

        select_btn = ttk.Button(file_frame, text="Browse...", command=self.select_lzw_file)
        select_btn.pack(side="left", padx=5, pady=5)

        self.decompress_file_label = ttk.Label(
            file_frame, text="No file selected", foreground="gray"
        )
        self.decompress_file_label.pack(side="left", padx=10, pady=5)

        # -- decompress button --
        self.decompress_btn = ttk.Button(
            self.decompress_tab,
            text="Decompress",
            command=self.do_decompress,
            state="disabled",
        )
        self.decompress_btn.pack(pady=5)

        # -- preview area --
        preview_frame = ttk.LabelFrame(self.decompress_tab, text="Step 2: Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # text preview
        self.decompress_text_preview = scrolledtext.ScrolledText(
            preview_frame, wrap="word", height=10, state="disabled"
        )

        # image preview
        self.decompress_image_label = tk.Label(preview_frame, bg="white")

        # placeholder
        self.decompress_preview_msg = ttk.Label(
            preview_frame,
            text="No file decompressed yet",
            foreground="gray",
            anchor="center",
        )
        self.decompress_preview_msg.pack(fill="both", expand=True)

        # -- save button --
        self.save_decompress_btn = ttk.Button(
            self.decompress_tab,
            text="Save Decompressed File",
            command=self.save_decompressed_file,
            state="disabled",
        )
        self.save_decompress_btn.pack(pady=5)

    # let user pick a .lzw file
    def select_lzw_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a .lzw file",
            filetypes=[("LZW Files", "*.lzw"), ("All Files", "*.*")],
        )
        # if cancelled skip
        if not file_path:
            return

        # save path
        self.decompress_file_path = file_path
        self.decompressed_data = None  # clear old stuff

        # show file name
        file_name = os.path.basename(file_path)
        self.decompress_file_label.configure(text=file_name, foreground="black")

        # enable decompress button
        self.decompress_btn.configure(state="normal")

        # reset preview
        self.decompress_text_preview.pack_forget()
        self.decompress_image_label.pack_forget()
        self.decompress_preview_msg.configure(text="Click 'Decompress' to proceed")
        self.decompress_preview_msg.pack(fill="both", expand=True)
        self.save_decompress_btn.configure(state="disabled")

    # run decompression and show result
    def do_decompress(self) -> None:
        if not self.decompress_file_path:
            messagebox.showwarning("Warning", "Please select a .lzw file first.")
            return

        try:
            # decompress using GUIUtils
            data, file_type, extension, _info = GUIUtils.decompress_lzw(self.decompress_file_path)

            # keep the data for saving later
            self.decompressed_data = data
            self.decompressed_type = file_type
            self.decompressed_ext = extension

            # hide everything first
            self.decompress_text_preview.pack_forget()
            self.decompress_image_label.pack_forget()
            self.decompress_preview_msg.pack_forget()

            if file_type == "text":
                # show text preview
                if isinstance(data, bytes):
                    text_content = data.decode("utf-8", errors="replace")
                else:
                    text_content = str(data)

                self.decompress_text_preview.configure(state="normal")
                self.decompress_text_preview.delete("1.0", "end")
                self.decompress_text_preview.insert("1.0", text_content[:10000])
                self.decompress_text_preview.configure(state="disabled")
                self.decompress_text_preview.pack(fill="both", expand=True, padx=5, pady=5)

            elif file_type == "image":
                # show image preview
                if isinstance(data, Image.Image):
                    display_image = data.copy()
                    display_image.thumbnail((400, 400))
                    self.decompress_photo = ImageTk.PhotoImage(display_image)
                    self.decompress_image_label.configure(image=self.decompress_photo)
                    self.decompress_image_label.pack(fill="both", expand=True, padx=5, pady=5)

            # enable save
            self.save_decompress_btn.configure(state="normal")
            messagebox.showinfo("Success", "Decompression completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Decompression failed:\n{e}")

    # save the decompressed result
    def save_decompressed_file(self) -> None:
        if self.decompressed_data is None:
            messagebox.showwarning("Warning", "No decompressed data to save.")
            return

        # suggest a name
        original_name = os.path.basename(self.decompress_file_path)
        base_name = os.path.splitext(original_name)[0]
        suggested_name = base_name + self.decompressed_ext

        # file type filters
        if self.decompressed_type == "text":
            filetypes = [
                ("Text Files", f"*{self.decompressed_ext}"),
                ("All Files", "*.*"),
            ]
        else:
            filetypes = [
                ("Image Files", f"*{self.decompressed_ext}"),
                ("All Files", "*.*"),
            ]

        file_path = filedialog.asksaveasfilename(
            title="Save decompressed file",
            defaultextension=self.decompressed_ext,
            initialfile=suggested_name,
            filetypes=filetypes,
        )
        # if cancelled skip
        if not file_path:
            return

        try:
            GUIUtils.save_decompressed_file(
                self.decompressed_data, self.decompressed_type, file_path
            )
            messagebox.showinfo("Saved", f"File saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    # start the gui
    def run(self) -> None:
        self.root.mainloop()


# run it
if __name__ == "__main__":
    app = LZWApp()
    app.run()
