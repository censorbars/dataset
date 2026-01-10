import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
import math
import cv2
import torch
import numpy as np
import logging
import json
import time
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, ImageSequence
from tqdm import tqdm
from nudenet import NudeDetector
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==================== CONSTANTS ====================

CENSOR_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_BREAST_EXPOSED"
}

LABEL_MAP = {
    "FEMALE_BREAST_EXPOSED": "exposed breasts",
    "FEMALE_GENITALIA_EXPOSED": "exposed genitalia",
    "BUTTOCKS_EXPOSED": "exposed buttocks",
    "BELLY_EXPOSED": "an exposed belly",
    "FACE_FEMALE": "a female face",
    "FACE_MALE": "a male face",
    "MALE_GENITALIA_EXPOSED": "exposed male genitalia",
    "ANUS_EXPOSED": "an exposed anus",
    "FEET_EXPOSED": "exposed feet",
    "FEMALE_BREAST_COVERED": "covered breasts",
    "FEMALE_GENITALIA_COVERED": "covered genitalia",
    "BELLY_COVERED": "a covered belly",
    "BUTTOCKS_COVERED": "covered buttocks",
    "MALE_BREAST_EXPOSED": "an exposed male chest",
    "ARMPITS_EXPOSED": "exposed armpits",
    "ARMPITS_COVERED": "covered armpits"
}

# Configuration constants with reasonable defaults
DEFAULT_EDGE_THRESHOLD = 5
DEFAULT_SAMPLE_INTERVAL_SECONDS = 2
DEFAULT_GIF_FRAME_LIMIT = 50
DEFAULT_CHECKPOINT_INTERVAL = 50
DEFAULT_VIDEO_MEMORY_CLEANUP_INTERVAL = 100


# ==================== DATA CLASSES ====================

@dataclass
class CensorStyle:
    """Configuration for censor bar appearance."""
    color: Tuple[int, int, int]  # RGB tuple
    opacity: int  # 0-255
    blur_radius: int  # 0 for solid bars, >0 for blur
    style: str  # 'solid', 'blur', 'pixelate'

    @classmethod
    def from_string(cls, color_str: str, opacity: int = 255, 
                    blur_radius: int = 0, style: str = 'solid') -> 'CensorStyle':
        """Create CensorStyle from color string (hex or name)."""
        color = cls._parse_color(color_str)
        return cls(color=color, opacity=opacity, blur_radius=blur_radius, style=style)

    @staticmethod
    def _parse_color(color_str: str) -> Tuple[int, int, int]:
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

        # Check if it's a named color
        if color_str in color_map:
            return color_map[color_str]

        # Parse hex color
        if color_str.startswith('#'):
            color_str = color_str[1:]

        if len(color_str) == 6:
            try:
                return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass

        raise ValueError(f"Invalid color format: {color_str}. Use hex (#RRGGBB) or color name.")


# ==================== LOGGING SETUP ====================

def setup_logging(debug: bool = False, log_file: str = 'processing.log') -> logging.Logger:
    """Configure logging with console and file handlers."""
    level = logging.DEBUG if debug else logging.INFO

    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter if not debug else detailed_formatter)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


# ==================== VALIDATION FUNCTIONS ====================

def validate_target_size(value: str) -> str:
    """Validate target size format (WxH)."""
    try:
        w, h = map(int, value.lower().split('x'))
        if w <= 0 or h <= 0:
            raise ValueError("Dimensions must be positive")
        if w > 8192 or h > 8192:
            raise ValueError("Dimensions too large (max 8192)")
        return f"{w}x{h}"
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid size format: {value}. Use WxH (e.g., 768x1024). {str(e)}"
        )


def validate_quality(value: str) -> int:
    """Validate quality parameter (1-100)."""
    try:
        quality = int(value)
        if not 1 <= quality <= 100:
            raise ValueError("Quality must be between 1 and 100")
        return quality
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Quality must be between 1 and 100, got: {value}"
        )


def validate_censor_color(value: str) -> str:
    """Validate censor color format."""
    try:
        CensorStyle._parse_color(value)
        return value
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def validate_opacity(value: str) -> int:
    """Validate opacity value (0-255)."""
    try:
        opacity = int(value)
        if not 0 <= opacity <= 255:
            raise ValueError("Opacity must be between 0 and 255")
        return opacity
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Opacity must be between 0 and 255, got: {value}"
        )


# ==================== CONFIGURATION ====================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        if config_file.suffix == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_file.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with comprehensive validation."""
    parser = argparse.ArgumentParser(
        description="Ultimate Intelligent Dataset Generator with Configurable Censoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python script.py --input ./raw --output ./dataset

  # With custom pink censor bars
  python script.py --input ./raw --output ./dataset --censor-color pink

  # With blur censoring
  python script.py --input ./raw --output ./dataset --censor-style blur --censor-blur-radius 20

  # With VLM captioning
  python script.py --input ./raw --output ./dataset --use-captioning

  # Update captions only
  python script.py --output ./dataset --update-captions-only --use-captioning

  # Resume interrupted processing
  python script.py --input ./raw --output ./dataset --resume

  # Load settings from config
  python script.py --config config.json
        """
    )

    # I/O Arguments
    parser.add_argument("--input", type=str, default="./raw",
                        help="Input raw folder containing images/videos/GIFs")
    parser.add_argument("--output", type=str, default="./dataset",
                        help="Output root folder for generated datasets")
    parser.add_argument("--config", type=str,
                        help="Path to JSON/YAML config file with default parameters")

    # Operation Modes
    parser.add_argument("--update-captions-only", action="store_true",
                        help="ONLY regenerate .txt files for existing images")
    parser.add_argument("--move-captions", type=str,
                        choices=['to-censored', 'to-uncensored', 'sync-both'],
                        help="Move/sync caption files between folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without processing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")

    # Processing Parameters
    parser.add_argument("--target-size", type=validate_target_size, default="768x1024",
                        help="Target image size in WxH format (e.g., 768x1024)")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="Minimum confidence score for detections (0.0-1.0)")
    parser.add_argument("--censor-box-resize", type=float, default=-0.3,
                        help="Resize censor box. Negative = smaller, Positive = larger")

    # Censor Bar Customization (NEW)
    censor_group = parser.add_argument_group('Censor Bar Customization')
    censor_group.add_argument("--censor-color", type=validate_censor_color, default="black",
                              help="Censor bar color (hex #RRGGBB or name: black, white, red, pink, etc.)")
    censor_group.add_argument("--censor-opacity", type=validate_opacity, default=255,
                              help="Censor bar opacity (0=transparent, 255=opaque)")
    censor_group.add_argument("--censor-style", type=str, 
                              choices=['solid', 'blur', 'pixelate'], default='solid',
                              help="Censor style: solid bar, blur, or pixelation")
    censor_group.add_argument("--censor-blur-radius", type=int, default=20,
                              help="Blur radius for blur style (pixels)")
    censor_group.add_argument("--censor-pixelate-size", type=int, default=16,
                              help="Pixel block size for pixelate style")

    # Output Settings
    parser.add_argument("--output-format", type=str, choices=['jpg', 'png', 'webp'],
                        default='jpg', help="Output image format")
    parser.add_argument("--output-quality", type=validate_quality, default=95,
                        help="Output image quality (1-100 for jpg/webp)")

    # Captioning
    parser.add_argument("--use-captioning", action="store_true",
                        help="Enable Florence-2 VLM description generation")
    parser.add_argument("--trigger-word", type=str, default="[trigger]",
                        help="Trigger token for AI toolkit training")

    # Video/GIF Processing
    parser.add_argument("--frame-similarity-threshold", type=int, default=8,
                        help="Max hamming distance for frame deduplication")
    parser.add_argument("--gif-frame-limit", type=int, default=DEFAULT_GIF_FRAME_LIMIT,
                        help="Maximum frames to extract from GIFs")
    parser.add_argument("--video-sample-interval", type=float, 
                        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
                        help="Seconds between video frame samples")

    # Checkpointing
    parser.add_argument("--checkpoint-file", type=str, 
                        default=".processing_checkpoint.json",
                        help="Path to checkpoint file for resume functionality")
    parser.add_argument("--checkpoint-interval", type=int,
                        default=DEFAULT_CHECKPOINT_INTERVAL,
                        help="Save checkpoint every N files")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference (cuda/cpu/mps)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose logs and visual debug images")
    parser.add_argument("--log-file", type=str, default="processing.log",
                        help="Path to log file")

    # First parse to check for config file
    args, remaining = parser.parse_known_args()

    # Load config file if provided and merge with CLI args
    if args.config:
        try:
            config = load_config(args.config)

            # Convert config keys to match argument names (handle underscores vs dashes)
            normalized_config = {}
            for key, value in config.items():
                normalized_key = key.replace('_', '-')
                normalized_config[normalized_key] = value

            # Set defaults from config (CLI args will override these)
            parser.set_defaults(**config)

            # Re-parse with config defaults applied
            args = parser.parse_args()

        except Exception as e:
            parser.error(f"Failed to load config: {e}")
    else:
        # Parse normally without config
        args = parser.parse_args()

    # Validate argument combinations
    mode_count = sum([
        args.update_captions_only,
        args.move_captions is not None,
        args.dry_run
    ])

    if mode_count > 1:
        parser.error("Only one operation mode can be specified at a time")

    # Validate thresholds
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0.0 and 1.0")

    if not -1.0 <= args.censor_box_resize <= 2.0:
        parser.error("--censor-box-resize must be between -1.0 and 2.0")

    return args


# ==================== UTILITY FUNCTIONS ====================

def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Calculate difference hash (dHash) for perceptual image comparison."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute dhash: {e}")
        return 0


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculate number of differing bits between two hashes."""
    return bin(hash1 ^ hash2).count('1')


@contextmanager
def atomic_write(filepath: Path):
    """Context manager for atomic file writing."""
    temp_path = filepath.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            yield f
        temp_path.replace(filepath)  # Atomic on POSIX systems
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


# ==================== ANALYZER ====================

class NudeNetAnalyzer:
    """Turns raw detections into natural human language."""

    def __init__(self, detections: List[Dict], width: int, height: int):
        self.dets = detections
        self.w = width
        self.h = height
        self.counts: Dict[str, int] = {}
        for d in detections:
            c = d['class']
            self.counts[c] = self.counts.get(c, 0) + 1

    def _get_people_count_and_gender(self) -> Tuple[int, str]:
        """Heuristic to estimate number of people and gender based on body parts."""
        faces_f = self.counts.get("FACE_FEMALE", 0)
        faces_m = self.counts.get("FACE_MALE", 0)

        breasts = (self.counts.get("FEMALE_BREAST_EXPOSED", 0) + 
                   self.counts.get("FEMALE_BREAST_COVERED", 0))
        buttocks = (self.counts.get("BUTTOCKS_EXPOSED", 0) + 
                    self.counts.get("BUTTOCKS_COVERED", 0))

        genitals_f = (self.counts.get("FEMALE_GENITALIA_EXPOSED", 0) + 
                      self.counts.get("FEMALE_GENITALIA_COVERED", 0))
        genitals_m = self.counts.get("MALE_GENITALIA_EXPOSED", 0)

        est_people_f = max(faces_f, genitals_f, math.ceil(breasts / 2))
        est_people_m = max(faces_m, genitals_m)

        if est_people_f == 0 and est_people_m == 0:
            if breasts > 0 or genitals_f > 0:
                est_people_f = 1
            elif genitals_m > 0:
                est_people_m = 1
            elif buttocks > 0:
                est_people_f = max(1, math.ceil(buttocks / 2))
            else:
                est_people_f = 1

        total = est_people_f + est_people_m

        if est_people_f > 0 and est_people_m == 0:
            gender_str = "1 woman" if est_people_f == 1 else f"{est_people_f} women"
        elif est_people_m > 0 and est_people_f == 0:
            gender_str = "1 man" if est_people_m == 1 else f"{est_people_m} men"
        else:
            gender_str = f"{est_people_f} women and {est_people_m} men"

        return total, gender_str

    def _get_nudity_state(self) -> str:
        """Determines if subjects are nude, topless, etc."""
        has_breasts = self.counts.get("FEMALE_BREAST_EXPOSED", 0) > 0
        has_genitals = (self.counts.get("FEMALE_GENITALIA_EXPOSED", 0) > 0 or
                        self.counts.get("MALE_GENITALIA_EXPOSED", 0) > 0)
        has_buttocks = self.counts.get("BUTTOCKS_EXPOSED", 0) > 0

        if has_genitals and has_breasts:
            return "fully nude"
        elif has_genitals:
            return "nude with exposed genitalia"
        elif has_breasts:
            return "topless"
        elif has_buttocks:
            return "showing exposed buttocks"
        else:
            return "partially nude"

    def _get_visible_features(self) -> List[str]:
        """Collect visible features for caption."""
        features = []

        priority_classes = [
            "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
            "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED",
            "BELLY_EXPOSED", "FEET_EXPOSED", "ARMPITS_EXPOSED"
        ]

        seen = set()
        for cls in priority_classes:
            if cls in self.counts and cls not in seen:
                readable = LABEL_MAP.get(cls, cls.lower().replace("_", " "))
                features.append(readable)
                seen.add(cls)

        return features

    def generate_smart_caption(self) -> str:
        """Generate intelligent caption based on detected features."""
        count, gender_str = self._get_people_count_and_gender()
        state = self._get_nudity_state()

        verb = 'is' if count == 1 else 'are'
        caption = f"This image shows {gender_str} who {verb} {state}."

        features = self._get_visible_features()
        if features and len(features) <= 4:
            if len(features) == 1:
                feat_str = features[0]
            else:
                feat_str = ", ".join(features[:-1]) + " and " + features[-1]
            caption += f" Visible features include {feat_str}."
        elif features:
            caption += f" The image shows {features[0]}, {features[1]} and other exposed areas."

        return caption


# ==================== CAPTION ENGINE ====================

class CaptionEngine:
    """Florence-2 VLM caption generation engine with proper resource management."""

    def __init__(self, device: str, args: Optional[argparse.Namespace] = None):
        self.args = args
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.device = device
        self.dtype: Optional[torch.dtype] = None
        self.logger = logging.getLogger(__name__)

        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers library not available. Captioning disabled.")
            return

        self.logger.info(f"Loading Florence-2 model on {device}...")
        try:
            self.dtype = torch.float16 if device == "cuda" else torch.float32

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
            self.logger.info("Florence-2 model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize VLM: {e}", 
                            exc_info=args.debug if args else False)
            self.model = None

    def describe(self, img: Image.Image) -> str:
        """Generate detailed caption for image."""
        if not self.model:
            return ""

        prompt = "<MORE_DETAILED_CAPTION>"
        try:
            inputs = self.processor(text=prompt, images=img, return_tensors="pt")

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
                image_size=(img.width, img.height)
            )

            text = parsed_answer.get(prompt, "").strip()

            if text.lower().startswith("the image shows"):
                text = text[15:].strip()

            self.logger.debug(f"Generated caption: {text[:100]}...")

            return text

        except Exception as e:
            self.logger.error(f"Florence-2 generation failed: {e}",
                            exc_info=self.args.debug if self.args else False)
            return ""

    def cleanup(self):
        """Release model resources."""
        if self.model is not None:
            self.logger.debug("Cleaning up caption engine resources...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# ==================== MAIN PIPELINE ====================

class Pipeline:
    """Main processing pipeline for dataset generation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.target_w, self.target_h = map(int, args.target_size.lower().split('x'))

        # Directory setup
        self.uncensored_dir = Path(args.output) / "dataset_uncensored"
        self.censored_dir = Path(args.output) / "dataset_censored"
        self.debug_dir = Path(args.output) / "debug_visuals"

        for d in [self.uncensored_dir, self.censored_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if args.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Initialize censor style
        self.censor_style = CensorStyle.from_string(
            args.censor_color,
            args.censor_opacity,
            args.censor_blur_radius,
            args.censor_style
        )
        self.logger.info(f"Censor style: {args.censor_style} with color {args.censor_color}")

        # Initialize detector
        self.logger.info("Loading NudeDetector...")
        try:
            self.detector = NudeDetector()
            self.logger.info("NudeDetector loaded successfully")
        except Exception as e:
            self.logger.critical(f"Failed to load NudeDetector: {e}")
            raise

        # Initialize captioner
        self.captioner: Optional[CaptionEngine] = None
        if args.use_captioning:
            try:
                self.captioner = CaptionEngine(args.device, args)
            except Exception as e:
                self.logger.error(f"Failed to initialize caption engine: {e}")

        # State management
        self.counter = 0
        self.processed_files: Set[str] = set()
        self.start_time: Optional[float] = None

        # Statistics
        self.stats = {
            'processed': 0,
            'skipped_no_targets': 0,
            'skipped_duplicate_frames': 0,
            'total_detections': 0,
            'censored_regions': 0,
            'errors': 0
        }

        # Load checkpoint if resuming
        if args.resume:
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Load processing progress from checkpoint file."""
        checkpoint_path = Path(self.args.checkpoint_file)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)

                self.processed_files = set(checkpoint.get('processed_files', []))
                self.counter = checkpoint.get('counter', 0)
                self.stats = checkpoint.get('stats', self.stats)

                self.logger.info(
                    f"Resumed from checkpoint: {len(self.processed_files)} "
                    f"files already processed"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    def _save_checkpoint(self):
        """Save processing progress to checkpoint file atomically."""
        try:
            checkpoint = {
                'processed_files': list(self.processed_files),
                'counter': self.counter,
                'stats': self.stats,
                'timestamp': time.time()
            }

            checkpoint_path = Path(self.args.checkpoint_file)
            with atomic_write(checkpoint_path) as f:
                json.dump(checkpoint, f, indent=2)

            self.logger.debug("Checkpoint saved")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def print_stats(self):
        """Print comprehensive statistics summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 70)
        print("📊 PROCESSING STATISTICS")
        print("=" * 70)
        print(f"✅ Processed frames:        {self.stats['processed']}")
        print(f"⏭️ Skipped (no targets):    {self.stats['skipped_no_targets']}")
        print(f"🔁 Skipped (duplicates):    {self.stats['skipped_duplicate_frames']}")
        print(f"🎯 Total detections:        {self.stats['total_detections']}")
        print(f"⬛ Censored regions:        {self.stats['censored_regions']}")
        print(f"❌ Errors encountered:      {self.stats['errors']}")
        print(f"💾 Images saved:            {self.counter}")
        print(f"⏱️ Total time:              {elapsed:.2f}s")
        if self.counter > 0:
            print(f"⚡ Average per image:       {elapsed / self.counter:.2f}s")
        print("=" * 70 + "\n")

    def xywh_to_xyxy_safe(self, box: List[float]) -> List[int]:
        """Converts NudeNet [x,y,w,h] to [x1,y1,x2,y2] and clamps to target."""
        x, y, w, h = box
        return [
            max(0, min(int(x), self.target_w)),
            max(0, min(int(y), self.target_h)),
            max(0, min(int(x + w), self.target_w)),
            max(0, min(int(y + h), self.target_h))
        ]

    def apply_scaling(self, box: List[int], scale_factor: float) -> List[int]:
        """Scales box from center with edge-aware logic."""
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

        edge_threshold = DEFAULT_EDGE_THRESHOLD

        touches_left = x1 <= edge_threshold
        touches_top = y1 <= edge_threshold
        touches_right = x2 >= (self.target_w - edge_threshold)
        touches_bottom = y2 >= (self.target_h - edge_threshold)

        if touches_left:
            new_x1 = 0
            if scale_factor < 0:
                new_x2 = x1 + new_w

        if touches_top:
            new_y1 = 0
            if scale_factor < 0:
                new_y2 = y1 + new_h

        if touches_right:
            new_x2 = self.target_w
            if scale_factor < 0:
                new_x1 = x2 - new_w

        if touches_bottom:
            new_y2 = self.target_h
            if scale_factor < 0:
                new_y1 = y2 - new_h

        return [
            max(0, min(int(new_x1), self.target_w)),
            max(0, min(int(new_y1), self.target_h)),
            max(0, min(int(new_x2), self.target_w)),
            max(0, min(int(new_y2), self.target_h))
        ]

    def apply_censor(self, img: Image.Image, box: List[int]) -> Image.Image:
        """Apply censoring based on configured style."""
        x1, y1, x2, y2 = box

        if self.censor_style.style == 'solid':
            draw = ImageDraw.Draw(img, 'RGBA')
            color_with_alpha = self.censor_style.color + (self.censor_style.opacity,)
            draw.rectangle([x1, y1, x2, y2], fill=color_with_alpha)

        elif self.censor_style.style == 'blur':
            # Extract region, blur it, paste back
            region = img.crop((x1, y1, x2, y2))
            region_cv = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
            blurred = cv2.GaussianBlur(
                region_cv,
                (self.censor_style.blur_radius * 2 + 1, 
                 self.censor_style.blur_radius * 2 + 1),
                0
            )
            blurred_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
            img.paste(blurred_pil, (x1, y1))

        elif self.censor_style.style == 'pixelate':
            # Extract region, shrink, enlarge
            region = img.crop((x1, y1, x2, y2))
            small_size = (
                max(1, (x2 - x1) // self.args.censor_pixelate_size),
                max(1, (y2 - y1) // self.args.censor_pixelate_size)
            )
            small = region.resize(small_size, Image.NEAREST)
            pixelated = small.resize((x2 - x1, y2 - y1), Image.NEAREST)
            img.paste(pixelated, (x1, y1))

        return img

    def manage_captions(self):
        """Move caption files between uncensored and censored folders."""
        mode = self.args.move_captions

        if mode == 'to-censored':
            source, target = self.uncensored_dir, self.censored_dir
            self.logger.info("Moving captions from uncensored → censored...")
        elif mode == 'to-uncensored':
            source, target = self.censored_dir, self.uncensored_dir
            self.logger.info("Moving captions from censored → uncensored...")
        elif mode == 'sync-both':
            self._sync_captions_bidirectional()
            return
        else:
            return

        txt_files = list(source.glob("*.txt"))
        moved = 0
        errors = 0

        for txt_file in tqdm(txt_files, desc="Moving captions"):
            target_file = target / txt_file.name
            try:
                if target_file.parent.exists():
                    shutil.move(str(txt_file), str(target_file))
                    moved += 1
                    self.logger.debug(f"Moved: {txt_file.name}")
            except Exception as e:
                errors += 1
                self.logger.error(f"Failed to move {txt_file.name}: {e}")

        self.logger.info(f"Completed: {moved} files moved, {errors} errors")

    def _sync_captions_bidirectional(self):
        """Ensure both folders have matching caption files."""
        self.logger.info("Syncing captions bidirectionally...")

        uncensored_txts = {f.stem: f for f in self.uncensored_dir.glob("*.txt")}
        censored_txts = {f.stem: f for f in self.censored_dir.glob("*.txt")}

        copied_to_censored = 0
        copied_to_uncensored = 0
        errors = 0

        for stem, src_file in tqdm(uncensored_txts.items(), desc="Syncing to censored"):
            if stem not in censored_txts:
                try:
                    shutil.copy2(str(src_file), str(self.censored_dir / src_file.name))
                    copied_to_censored += 1
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Failed to copy {src_file.name}: {e}")

        for stem, src_file in tqdm(censored_txts.items(), desc="Syncing to uncensored"):
            if stem not in uncensored_txts:
                try:
                    shutil.copy2(str(src_file), str(self.uncensored_dir / src_file.name))
                    copied_to_uncensored += 1
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Failed to copy {src_file.name}: {e}")

        self.logger.info(
            f"Sync complete: {copied_to_censored} → censored, "
            f"{copied_to_uncensored} → uncensored, {errors} errors"
        )

    def process_data(self):
        """Main entry point for processing."""
        self.start_time = time.time()

        try:
            if self.args.move_captions:
                self.manage_captions()
                return

            if self.args.update_captions_only:
                self.run_caption_update()
                return

            if self.args.dry_run:
                self.run_dry_run()
                return

            self.run_full_processing()

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            self.logger.critical(f"Fatal error during processing: {e}", exc_info=True)
            self._save_checkpoint()
            raise
        finally:
            # Cleanup resources
            if self.captioner:
                self.captioner.cleanup()

            self.print_stats()

            # Cleanup checkpoint if completed successfully
            if not self.args.resume and Path(self.args.checkpoint_file).exists():
                try:
                    Path(self.args.checkpoint_file).unlink()
                    self.logger.debug("Checkpoint file removed")
                except Exception as e:
                    self.logger.debug(f"Could not remove checkpoint: {e}")

    def run_dry_run(self):
        """Preview what would be processed without actually processing."""
        input_path = Path(self.args.input)
        files = sorted([
            f for f in input_path.rglob("*")
            if f.is_file() and not f.name.startswith('.')
        ])

        print(f"\n🔍 DRY RUN MODE - No files will be modified")
        print(f"📁 Input folder: {input_path}")
        print(f"📁 Output folder: {self.args.output}")
        print(f"📊 Found {len(files)} files\n")

        file_types: Dict[str, int] = defaultdict(int)
        for f in files:
            ext = f.suffix.lower()
            if ext in {'.jpg', '.jpeg', '.png', '.webp'}:
                file_types['images'] += 1
            elif ext in {'.mp4', '.avi', '.mov', '.webm'}:
                file_types['videos'] += 1
            elif ext == '.gif':
                file_types['gifs'] += 1
            else:
                file_types['unknown'] += 1

        print("File breakdown:")
        for ftype, count in file_types.items():
            print(f"  - {ftype.capitalize()}: {count}")

        print(f"\n⚙️  Settings:")
        print(f"  - Target size: {self.args.target_size}")
        print(f"  - Score threshold: {self.args.score_threshold}")
        print(f"  - Censor box resize: {self.args.censor_box_resize}")
        print(f"  - Censor style: {self.args.censor_style}")
        print(f"  - Censor color: {self.args.censor_color}")
        print(f"  - Censor opacity: {self.args.censor_opacity}")
        print(f"  - Output format: {self.args.output_format}")
        print(f"  - Output quality: {self.args.output_quality}")
        print(f"  - VLM captioning: {'Enabled' if self.args.use_captioning else 'Disabled'}")
        print(f"  - Frame similarity: {self.args.frame_similarity_threshold}")

    def run_caption_update(self):
        """Update captions for existing images."""
        self.logger.info("MODE: Updating Captions Only...")

        # Find all images in uncensored directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        files = []
        for ext in image_extensions:
            files.extend(self.uncensored_dir.glob(ext))

        files = sorted(files)
        self.logger.info(f"Found {len(files)} existing images")

        errors = []

        for f in tqdm(files, desc="Updating captions"):
            try:
                img = Image.open(f).convert("RGB")

                # Run detection
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                detections = self.detector.detect(img_bgr)

                # Filter valid detections
                valid_dets = []
                for d in detections:
                    if d['score'] >= self.args.score_threshold:
                        d['box'] = self.xywh_to_xyxy_safe(d['box'])
                        valid_dets.append(d)

                # Generate smart caption
                analyzer = NudeNetAnalyzer(valid_dets, img.width, img.height)
                smart_cap = analyzer.generate_smart_caption()

                # Generate VLM caption if enabled
                vlm_cap = ""
                if self.captioner:
                    self.logger.debug(f"Running Florence-2 for {f.name}...")
                    vlm_cap = self.captioner.describe(img)

                # Choose caption
                if vlm_cap:
                    final_caption = f"{self.args.trigger_word} {vlm_cap}"
                    self.logger.debug(f"Using VLM caption for {f.name}")
                else:
                    final_caption = f"{self.args.trigger_word} {smart_cap}"
                    self.logger.debug(f"Using Smart caption for {f.name}")

                # Save caption to both directories
                txt_uncensored = f.with_suffix(".txt")
                # txt_censored = self.censored_dir / txt_uncensored.name

                with open(txt_uncensored, "w", encoding="utf-8") as tf:
                    tf.write(final_caption)

                # with open(txt_censored, "w", encoding="utf-8") as tf:
                #     tf.write(final_caption)

                self.logger.debug(
                    f"Saved: {txt_uncensored.name} | Length: {len(final_caption)} chars"
                )
                self.stats['processed'] += 1

            except Exception as e:
                errors.append((f.name, str(e)))
                self.stats['errors'] += 1
                self.logger.error(f"Error updating {f.name}: {e}")

        if errors:
            self.logger.warning(f"Failed to update {len(errors)} captions")
            for fname, error in errors[:10]:
                self.logger.debug(f"  - {fname}: {error}")

    def run_full_processing(self):
        """Process all input files."""
        input_path = Path(self.args.input)
        files = sorted([
            f for f in input_path.rglob("*")
            if f.is_file() and not f.name.startswith('.')
        ])

        self.logger.info(f"Processing {len(files)} files from {input_path}")

        if self.args.resume:
            files = [f for f in files if str(f) not in self.processed_files]
            self.logger.info(f"Resuming: {len(files)} files remaining")

        errors = []

        for f in tqdm(files, desc="Processing files"):
            try:
                ext = f.suffix.lower()
                stem = f.stem

                if ext in {'.jpg', '.jpeg', '.png', '.webp'}:
                    img = Image.open(f).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    self.process_frame(img, stem)
                elif ext in {'.mp4', '.avi', '.mov', '.webm'}:
                    self.process_video_intelligent(f)
                elif ext == ".gif":
                    self.process_gif_intelligent(f)
                else:
                    self.logger.debug(f"Skipping unsupported file type: {f.name}")
                    continue

                self.processed_files.add(str(f))

                # Periodic checkpoint saving
                if len(self.processed_files) % self.args.checkpoint_interval == 0:
                    self._save_checkpoint()

            except (IOError, OSError) as e:
                errors.append((f.name, f"IO Error: {e}"))
                self.stats['errors'] += 1
                self.logger.error(f"IO Error processing {f.name}: {e}")
            except cv2.error as e:
                errors.append((f.name, f"OpenCV Error: {e}"))
                self.stats['errors'] += 1
                self.logger.error(f"OpenCV Error processing {f.name}: {e}")
            except Exception as e:
                errors.append((f.name, f"Unexpected error: {e}"))
                self.stats['errors'] += 1
                self.logger.error(
                    f"Unexpected error processing {f.name}: {e}",
                    exc_info=self.args.debug
                )

        self._save_checkpoint()

        if errors:
            self.logger.warning(f"Encountered {len(errors)} errors during processing")
            if self.args.debug:
                for fname, error in errors[:20]:
                    self.logger.debug(f"  - {fname}: {error}")

    def process_video_intelligent(self, video_path: Path):
        """Intelligently extract unique frames from video using perceptual hashing."""
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sample_interval = max(1, int(fps * self.args.video_sample_interval))

            prev_hash = None
            frame_idx = 0
            extracted = 0

            self.logger.debug(
                f"Video: {video_path.name} | FPS: {fps:.1f} | "
                f"Total frames: {total_frames} | Sample interval: {sample_interval}"
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    current_hash = dhash(frame)

                    if (prev_hash is None or 
                        hamming_distance(current_hash, prev_hash) > 
                        self.args.frame_similarity_threshold):

                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self.process_frame(img, f"{video_path.stem}_f{frame_idx:05d}")
                        prev_hash = current_hash
                        extracted += 1
                    else:
                        self.stats['skipped_duplicate_frames'] += 1

                frame_idx += 1

                # Periodic memory cleanup
                if frame_idx % DEFAULT_VIDEO_MEMORY_CLEANUP_INTERVAL == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            cap.release()
            self.logger.debug(f"Extracted {extracted} unique frames from {video_path.name}")

        except Exception as e:
            self.logger.error(
                f"Failed to process video {video_path.name}: {e}",
                exc_info=self.args.debug
            )
            raise

    def process_gif_intelligent(self, gif_path: Path):
        """Intelligently extract unique frames from GIF using perceptual hashing."""
        try:
            g = Image.open(gif_path)
            prev_hash = None
            extracted = 0

            for i, frame in enumerate(ImageSequence.Iterator(g)):
                if i >= self.args.gif_frame_limit:
                    self.logger.debug(
                        f"Reached GIF frame limit ({self.args.gif_frame_limit}) "
                        f"for {gif_path.name}"
                    )
                    break

                frame_rgb = frame.convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
                current_hash = dhash(frame_bgr)

                if (prev_hash is None or 
                    hamming_distance(current_hash, prev_hash) > 
                    self.args.frame_similarity_threshold):

                    self.process_frame(frame_rgb, f"{gif_path.stem}_g{i:03d}")
                    prev_hash = current_hash
                    extracted += 1
                else:
                    self.stats['skipped_duplicate_frames'] += 1

            self.logger.debug(f"Extracted {extracted} unique frames from {gif_path.name}")

        except Exception as e:
            self.logger.error(
                f"Failed to process GIF {gif_path.name}: {e}",
                exc_info=self.args.debug
            )
            raise

    def process_frame(self, pil_img: Image.Image, source_name: str):
        """Process a single frame/image."""
        try:
            # Resize to target
            img = ImageOps.fit(
                pil_img,
                (self.target_w, self.target_h),
                method=Image.LANCZOS,
                centering=(0.5, 0.5)
            )

            # Run detection
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            detections = self.detector.detect(img_bgr)

            # Filter and collect valid detections
            valid_dets = []
            censor_targets = []

            for d in detections:
                if d['score'] < self.args.score_threshold:
                    continue

                xyxy = self.xywh_to_xyxy_safe(d['box'])
                d['box'] = xyxy
                valid_dets.append(d)
                self.stats['total_detections'] += 1

                if d['class'] in CENSOR_CLASSES:
                    censor_targets.append({
                        'box': self.apply_scaling(xyxy, self.args.censor_box_resize),
                        'label': d['class']
                    })
                    self.stats['censored_regions'] += 1

            # Skip if no targets
            if not censor_targets:
                self.stats['skipped_no_targets'] += 1
                return

            # Generate captions
            analyzer = NudeNetAnalyzer(valid_dets, self.target_w, self.target_h)
            smart_caption = analyzer.generate_smart_caption()

            vlm_text = ""
            if self.captioner:
                vlm_text = self.captioner.describe(img)

            if vlm_text:
                final_caption = f"{self.args.trigger_word} {vlm_text}"
            else:
                final_caption = f"{self.args.trigger_word} {smart_caption}"

            # Create censored version with custom style
            censored_img = img.copy()
            for t in censor_targets:
                censored_img = self.apply_censor(censored_img, t['box'])

            # Debug visualization
            if self.args.debug:
                self._create_debug_visualization(
                    img, valid_dets, censor_targets, self.counter, final_caption
                )

            # Save outputs
            fname = f"{self.counter:06d}"
            self.counter += 1
            self.stats['processed'] += 1

            # Determine file extension and save parameters
            ext = self.args.output_format
            save_params: Dict[str, Any] = {}

            if ext == 'jpg':
                save_params = {'quality': self.args.output_quality, 'optimize': True}
            elif ext == 'webp':
                save_params = {
                    'quality': self.args.output_quality,
                    'method': 6
                }
            elif ext == 'png':
                save_params = {'optimize': True}

            # Save uncensored
            uncensored_img_path = self.uncensored_dir / f"{fname}.{ext}"
            img.save(uncensored_img_path, **save_params)

            # Save censored
            censored_img_path = self.censored_dir / f"{fname}.{ext}"
            censored_img.save(censored_img_path, **save_params)

            # Save captions to both directories
            uncensored_txt = self.uncensored_dir / f"{fname}.txt"
            # censored_txt = self.censored_dir / f"{fname}.txt"

            with open(uncensored_txt, "w", encoding="utf-8") as tf:
                tf.write(final_caption)

            # with open(censored_txt, "w", encoding="utf-8") as tf:
            #     tf.write(final_caption)

            self.logger.debug(
                f"Saved {fname}.{ext} | Censored: {len(censor_targets)} regions | "
                f"Caption: {len(final_caption)} chars"
            )

        except Exception as e:
            self.logger.error(f"Failed to process frame {source_name}: {e}",
                            exc_info=self.args.debug)
            raise

    def _create_debug_visualization(self, img: Image.Image, 
                                detections: List[Dict],
                                censor_targets: List[Dict],
                                counter: int,
                                caption: str):
        """Create compact debug visualization with essential info."""
        try:
            from PIL import ImageFont

            # Create debug image
            debug_img = img.copy()
            draw = ImageDraw.Draw(debug_img)

            # Try to load a font (fallback to default if not available)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()

            # Draw all detections with labels
            for d in detections:
                x1, y1, x2, y2 = d['box']
                cls = d['class']
                score = d['score']

                # Green boxes for all detections
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

                # Compact label with class and score
                label = f"{cls} {score:.2f}"

                # Background for text
                text_bbox = draw.textbbox((x1, y1 - 12), label, font=font)
                draw.rectangle(text_bbox, fill="green")
                draw.text((x1, y1 - 12), label, fill="white", font=font)

            # Draw censor targets with labels
            for t in censor_targets:
                x1, y1, x2, y2 = t['box']
                label_text = t['label']

                # Red boxes for censor targets (thicker)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

                # Compact label
                label = f"CENSOR: {label_text}"
                text_bbox = draw.textbbox((x1, y2 + 2), label, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y2 + 2), label, fill="white", font=font)

            # Compact one-line info bar at bottom
            info_text = (
                f"#{counter:06d} | "
                f"Det:{len(detections)} Cen:{len(censor_targets)} | "
                f"{img.width}x{img.height} | "
                f"{self.args.censor_style}/{self.args.censor_color} | "
                f"Resize:{self.args.censor_box_resize}"
            )

            # Draw compact info bar at bottom
            text_bbox = draw.textbbox((0, img.height - 15), info_text, font=font)
            bar_height = text_bbox[3] - text_bbox[1] + 4
            draw.rectangle([0, img.height - bar_height, img.width, img.height], 
                        fill="black", outline="yellow", width=1)
            draw.text((2, img.height - bar_height + 2), info_text, fill="yellow", font=font)

            # Save with counter-based filename
            debug_path = self.debug_dir / f"{counter:06d}.jpg"
            debug_img.save(debug_path, quality=95)

            self.logger.debug(f"Debug visualization saved: {debug_path.name}")

        except Exception as e:
            self.logger.warning(f"Failed to create debug visualization: {e}", 
                            exc_info=self.args.debug)

    def cleanup(self):
        """Clean up resources."""
        if self.captioner:
            self.captioner.cleanup()


# ==================== MAIN ====================

def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logging(debug=args.debug, log_file=args.log_file)

    logger.info("="*70)
    logger.info("Ultimate Intelligent Dataset Generator")
    logger.info("="*70)
    logger.info(f"Mode: {args.move_captions or 'Full Processing'}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Censor Style: {args.censor_style} ({args.censor_color})")
    logger.info("="*70)

    try:
        pipeline = Pipeline(args)
        pipeline.process_data()
        logger.info("Processing completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
