# Watermarker

GhostMark is a Python web application that allows users to add watermarks to images and videos.

## Features

- Add watermarks to images and videos using image watermarks (including PNG, JPEG, GIF, and ICO files)
- Advanced watermarking technique:
  - Combined (LSB + DCT + Deep Learning) technique for maximum robustness
- Live previews of uploaded media and processing results
- Control watermark size for image watermarks (percentage of media width)
- Two visibility options:
  - Visible watermarks clearly show on your media
  - Invisible watermarks using combined techniques for optimal security
- Simple toggle switches for easy option selection
- Extract watermarks from previously watermarked files
- Modern, responsive web interface using Flask, HTML, CSS, and JavaScript.
- Real-time progress tracking for video processing.

## Installation

1. Clone the repository:

```
git clone https://github.com/TrentPierce/watermarker.git
```

2. Install the required dependencies:

```
pip install pillow opencv-python-headless flask numpy
```

## Usage

1. Run the script:

```
python watermarker.py
```

2. Open your browser and navigate to `http://localhost:5000`

### Adding Watermarks

1. Upload a file using the file selector.
2. Preview your uploaded media directly in the web interface.
3. Upload a watermark image (transparent PNG recommended)
4. Adjust the size using the slider
5. Choose between visible or invisible watermarking.
6. Select a position for the watermark.
7. Click the "Add Watermark" button.
8. Wait for the processing to complete.
9. Preview the watermarked result before downloading.
10. Download the watermarked file.

### Notes on Image Watermarks

- Transparent PNG files work best for watermarking
- GIF watermarks will use only the first frame
- For invisible image watermarks, a very light version of the image is embedded
- The watermark size can be adjusted from 5% to 50% of the media width

### Extracting Watermarks

1. Go to the "Extract Watermark" tab at the top of the page.
2. Upload a file that contains a watermark.
3. Click "Extract Watermark".
4. The system will process your file and show the extracted watermark.
5. You can download the extracted watermark.

## Watermarking Information

Watermarker uses a combined watermarking technique that incorporates:

1. **LSB (Least Significant Bit)**: A spatial domain technique that hides watermark data in the least significant bits of pixel values.

2. **DCT (Discrete Cosine Transform)**: This frequency domain technique embeds watermark data in the mid-frequency coefficients of the image after applying a DCT. This method is more robust against image processing operations like compression.

3. **Deep Learning Approach**: A neural network-inspired approach that creates imperceptible and robust watermarks that are resistant to various image transformations.

By combining these three approaches, Watermarker offers the most robust protection for your media. The watermark is invisible to the human eye but can be detected and extracted using the application.

