#!/usr/bin/env python3
"""
Florence-2 VLM Caption Generator
Generates caption .txt files for all images in a folder using Florence-2 Vision Language Model.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError:
    print("ERROR: transformers library not found. Install with: pip install transformers")
    sys.exit(1)


class Florence2Captioner:
    """Florence-2 VLM caption generation engine."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Loading Florence-2 model on {device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="eager"
            ).to(device)

            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True
            )
            print("Florence-2 model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load Florence-2 model: {e}")
            sys.exit(1)

    def generate_caption(self, image: Image.Image) -> str:
        """Generate detailed caption for an image."""
        prompt = "<MORE_DETAILED_CAPTION>"

        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            inputs = {
                k: v.to(self.device).to(self.dtype) if k == "pixel_values"
                else v.to(self.device)
                for k, v in inputs.items()
            }

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                use_cache=False
            )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            caption = parsed_answer.get(prompt, "").strip()

            # Remove redundant prefix if present
            if caption.lower().startswith("the image shows"):
                caption = caption[15:].strip()

            return caption

        except Exception as e:
            print(f"WARNING: Failed to generate caption: {e}")
            return ""

    def cleanup(self):
        """Release model resources."""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate caption .txt files for images using Florence-2 VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python capt.py --input-dir ./my_images --trigger-word "photo,"
  python capt.py --input-dir ./dataset --trigger-word "[trigger]," --lowercase
  python capt.py --input-dir ./photos --trigger-word "image of" --device cpu
        """
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing images to caption"
    )

    parser.add_argument(
        "--trigger-word",
        type=str,
        default="",
        help="Text to prepend to each caption (e.g., '[trigger],' or 'photo,')"
    )

    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert the first letter of the caption to lowercase"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for model inference (default: cuda)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files (default: skip existing)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(image_files)} image(s) in {input_dir}")

    # Check for existing caption files
    if not args.overwrite:
        image_files = [f for f in image_files if not (f.with_suffix(".txt").exists())]
        print(f"{len(image_files)} image(s) need captions (skipping existing)")

    if not image_files:
        print("All images already have captions. Use --overwrite to regenerate.")
        sys.exit(0)

    # Initialize captioner
    captioner = Florence2Captioner(device=args.device)

    # Process images
    successful = 0
    failed = 0

    try:
        for image_path in tqdm(image_files, desc="Generating captions"):
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")

                # Generate caption
                caption = captioner.generate_caption(image)

                if not caption:
                    print(f"WARNING: Empty caption for {image_path.name}, skipping")
                    failed += 1
                    continue

                # Apply lowercase if requested
                if args.lowercase and caption:
                    caption = caption[0].lower() + caption[1:] if len(caption) > 1 else caption.lower()

                # Prepend trigger word
                if args.trigger_word:
                    # Ensure proper spacing
                    trigger = args.trigger_word
                    if not trigger.endswith(" ") and caption:
                        trigger += " "
                    caption = trigger + caption

                # Save caption to .txt file
                txt_path = image_path.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                successful += 1

            except Exception as e:
                print(f"ERROR processing {image_path.name}: {e}")
                failed += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        captioner.cleanup()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Caption generation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
