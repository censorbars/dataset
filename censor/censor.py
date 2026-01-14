#!/usr/bin/env python3
"""
Image Censoring Tool
Applies censor bars to images based on nudity detection.

Usage:
    python censor.py --input images --output censored
    python censor.py --input images --output censored --color pink --style blur
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from nudenet import NudeDetector


# ==================== CONFIGURATION ====================

CENSOR_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_BREAST_EXPOSED"
}


@dataclass
class CensorStyle:
    """Configuration for censor bar appearance."""
    color: Tuple[int, int, int]
    opacity: int
    blur_radius: int
    style: str

    @staticmethod
    def parse_color(color_str: str) -> Tuple[int, int, int]:
        """Parse color from hex string or common color names."""
        color_map = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'pink': (255, 192, 203),
            'purple': (128, 0, 128),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }

        color_str = color_str.lower().strip()

        if color_str in color_map:
            return color_map[color_str]

        if color_str.startswith('#'):
            color_str = color_str[1:]

        if len(color_str) == 6:
            try:
                return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass

        raise ValueError(f"Invalid color: {color_str}. Use hex (#RRGGBB) or name.")


# ==================== ARGUMENT PARSING ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply censor bars to images based on nudity detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with black bars
  python censor.py --input images --output censored

  # Pink solid bars
  python censor.py --input images --output censored --color pink

  # Blur censoring
  python censor.py --input images --output censored --style blur --blur-radius 30

  # Pixelated censoring
  python censor.py --input images --output censored --style pixelate --pixel-size 20

  # Custom color with transparency
  python censor.py --input images --output censored --color "#FF1493" --opacity 200
        """
    )

    parser.add_argument("--input", type=str, required=True,
                       help="Input folder containing images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output folder for censored images")

    # Censor customization
    parser.add_argument("--color", type=str, default="black",
                       help="Censor bar color (hex #RRGGBB or name: black, white, red, pink, etc.)")
    parser.add_argument("--opacity", type=int, default=255,
                       help="Censor bar opacity (0=transparent, 255=opaque)")
    parser.add_argument("--style", type=str, choices=['solid', 'blur', 'pixelate'],
                       default='solid', help="Censor style")
    parser.add_argument("--blur-radius", type=int, default=20,
                       help="Blur radius for blur style (pixels)")
    parser.add_argument("--pixel-size", type=int, default=16,
                       help="Pixel block size for pixelate style")

    # Detection parameters
    parser.add_argument("--threshold", type=float, default=0.35,
                       help="Detection confidence threshold (0.0-1.0)")
    parser.add_argument("--box-resize", type=float, default=-0.3,
                       help="Resize censor box. Negative=smaller, Positive=larger")

    # Output settings
    parser.add_argument("--format", type=str, choices=['jpg', 'png', 'webp'],
                       default='jpg', help="Output image format")
    parser.add_argument("--quality", type=int, default=95,
                       help="Output quality for jpg/webp (1-100)")

    args = parser.parse_args()

    # Validate
    if not 0 <= args.opacity <= 255:
        parser.error("--opacity must be between 0 and 255")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    try:
        CensorStyle.parse_color(args.color)
    except ValueError as e:
        parser.error(str(e))

    return args


# ==================== IMAGE PROCESSING ====================

class ImageCensor:
    """Main class for applying censor bars to images."""

    def __init__(self, args):
        self.args = args

        # Setup censor style
        self.censor_style = CensorStyle(
            color=CensorStyle.parse_color(args.color),
            opacity=args.opacity,
            blur_radius=args.blur_radius,
            style=args.style
        )

        # Initialize detector
        print("Loading NudeDetector...")
        self.detector = NudeDetector()
        print("✓ NudeDetector loaded")

        # Setup directories
        self.input_dir = Path(args.input)
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'censored_regions': 0
        }

    def xywh_to_xyxy(self, box: List[float], img_width: int, img_height: int) -> List[int]:
        """Convert [x,y,w,h] to [x1,y1,x2,y2] and clamp to image bounds."""
        x, y, w, h = box
        return [
            max(0, min(int(x), img_width)),
            max(0, min(int(y), img_height)),
            max(0, min(int(x + w), img_width)),
            max(0, min(int(y + h), img_height))
        ]

    def apply_box_resize(self, box: List[int], scale_factor: float, 
                        img_width: int, img_height: int) -> List[int]:
        """Resize box from center."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2

        new_w = w * (1.0 + scale_factor)
        new_h = h * (1.0 + scale_factor)

        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2

        return [
            max(0, min(int(new_x1), img_width)),
            max(0, min(int(new_y1), img_height)),
            max(0, min(int(new_x2), img_width)),
            max(0, min(int(new_y2), img_height))
        ]

    def apply_censor(self, img: Image.Image, box: List[int]) -> Image.Image:
        """Apply censoring based on configured style."""
        x1, y1, x2, y2 = box

        if self.censor_style.style == 'solid':
            draw = ImageDraw.Draw(img, 'RGBA')
            color_with_alpha = self.censor_style.color + (self.censor_style.opacity,)
            draw.rectangle([x1, y1, x2, y2], fill=color_with_alpha)

        elif self.censor_style.style == 'blur':
            region = img.crop((x1, y1, x2, y2))
            region_cv = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
            kernel_size = self.censor_style.blur_radius * 2 + 1
            blurred = cv2.GaussianBlur(region_cv, (kernel_size, kernel_size), 0)
            blurred_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
            img.paste(blurred_pil, (x1, y1))

        elif self.censor_style.style == 'pixelate':
            region = img.crop((x1, y1, x2, y2))
            small_size = (
                max(1, (x2 - x1) // self.args.pixel_size),
                max(1, (y2 - y1) // self.args.pixel_size)
            )
            small = region.resize(small_size, Image.NEAREST)
            pixelated = small.resize((x2 - x1, y2 - y1), Image.NEAREST)
            img.paste(pixelated, (x1, y1))

        return img

    def process_image(self, image_path: Path):
        """Process a single image."""
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)

            # Run detection
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detections = self.detector.detect(img_bgr)

            # Filter and collect censor targets
            censor_targets = []
            for d in detections:
                if d['score'] < self.args.threshold:
                    continue

                if d['class'] in CENSOR_CLASSES:
                    xyxy = self.xywh_to_xyxy(d['box'], img.width, img.height)
                    resized_box = self.apply_box_resize(
                        xyxy, self.args.box_resize, img.width, img.height
                    )
                    censor_targets.append(resized_box)

            # Skip if no targets
            if not censor_targets:
                self.stats['skipped'] += 1
                return

            # Apply censoring
            censored_img = img.copy()
            for box in censor_targets:
                censored_img = self.apply_censor(censored_img, box)
                self.stats['censored_regions'] += 1

            # Save output
            output_path = self.output_dir / f"{image_path.stem}.{self.args.format}"

            save_params = {}
            if self.args.format == 'jpg':
                save_params = {'quality': self.args.quality, 'optimize': True}
            elif self.args.format == 'webp':
                save_params = {'quality': self.args.quality, 'method': 6}
            elif self.args.format == 'png':
                save_params = {'optimize': True}

            censored_img.save(output_path, **save_params)
            self.stats['processed'] += 1

        except Exception as e:
            self.stats['errors'] += 1
            print(f"Error processing {image_path.name}: {e}")

    def process_all(self):
        """Process all images in input directory."""
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(ext))
            image_files.extend(self.input_dir.glob(ext.upper()))

        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No images found in {self.input_dir}")
            return

        print(f"\nFound {len(image_files)} images")
        print(f"Censor style: {self.args.style} ({self.args.color})")
        print(f"Output format: {self.args.format}\n")

        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            self.process_image(img_path)

        # Print statistics
        self.print_stats()

    def print_stats(self):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"✅ Images processed:     {self.stats['processed']}")
        print(f"⏭️  Skipped (no nudity):  {self.stats['skipped']}")
        print(f"⬛ Censored regions:     {self.stats['censored_regions']}")
        print(f"❌ Errors:               {self.stats['errors']}")
        print(f"📁 Output directory:     {self.output_dir}")
        print("="*60)


# ==================== MAIN ====================

def main():
    print("="*60)
    print("IMAGE CENSORING TOOL")
    print("="*60)

    args = parse_args()

    # Validate input directory
    if not Path(args.input).exists():
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)

    try:
        censor = ImageCensor(args)
        censor.process_all()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
