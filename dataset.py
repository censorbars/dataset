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

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Constants
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


def setup_logging(debug=False, log_file='processing.log'):
    """Configure logging with console and file handlers."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter if not debug else detailed_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


def validate_target_size(value):
    """Validate target size format (WxH)."""
    try:
        w, h = map(int, value.lower().split('x'))
        if w <= 0 or h <= 0:
            raise ValueError("Dimensions must be positive")
        return f"{w}x{h}"
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid size format: {value}. Use WxH (e.g., 768x1024)"
        )


def validate_quality(value):
    """Validate quality parameter (1-100)."""
    try:
        quality = int(value)
        if not 1 <= quality <= 100:
            raise ValueError
        return quality
    except:
        raise argparse.ArgumentTypeError(
            f"Quality must be between 1 and 100, got: {value}"
        )


def load_config(config_path):
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
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")


def parse_args():
    """Parse command-line arguments with comprehensive validation."""
    parser = argparse.ArgumentParser(
        description="Ultimate Intelligent Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python script.py --input ./raw --output ./dataset

  # With VLM captioning
  python script.py --input ./raw --output ./dataset --use-captioning

  # Update captions only
  python script.py --output ./dataset --update-captions-only --use-captioning

  # Move captions from uncensored to censored
  python script.py --output ./dataset --move-captions to-censored

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
                        help="ONLY regenerate .txt files for existing uncensored images")
    parser.add_argument("--move-captions", type=str, 
                        choices=['to-censored', 'to-uncensored', 'sync-both'],
                        help="Move caption files between folders: "
                             "'to-censored' moves from uncensored to censored, "
                             "'to-uncensored' moves from censored to uncensored, "
                             "'sync-both' ensures both folders have matching captions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without actually processing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (skips already processed files)")
    
    # Processing Parameters
    parser.add_argument("--target-size", type=validate_target_size, default="768x1024",
                        help="Target image size in WxH format (e.g., 768x1024)")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="Minimum confidence score for detections (0.0-1.0)")
    parser.add_argument("--box-scale", type=float, default=-0.3,
                        help="Resize black box. Negative = smaller, Positive = larger "
                             "(e.g., -0.3 = 30%% smaller, 0.3 = 30%% larger)")
    
    # Output Settings
    parser.add_argument("--output-format", type=str, choices=['jpg', 'png', 'webp'],
                        default='jpg', help="Output image format")
    parser.add_argument("--output-quality", type=validate_quality, default=95,
                        help="Output image quality (1-100 for jpg/webp)")
    
    # Captioning
    parser.add_argument("--use-captioning", action="store_true",
                        help="Enable Florence-2 VLM description generation")
    parser.add_argument("--trigger-word", type=str, default="[trigger]",
                        help="Trigger token for AI toolkit training (placed at caption start)")
    
    # Video/GIF Processing
    parser.add_argument("--frame-similarity-threshold", type=int, default=8,
                        help="Max hamming distance for frame deduplication (lower = more strict)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of frames to process in parallel (experimental)")
    
    # Checkpointing
    parser.add_argument("--checkpoint-file", type=str, default=".processing_checkpoint.json",
                        help="Path to checkpoint file for resume functionality")
    
    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference (cuda/cpu/mps)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose logs and visual debug images")
    parser.add_argument("--log-file", type=str, default="processing.log",
                        help="Path to log file")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        try:
            config = load_config(args.config)
            # Update args with config values (CLI args take precedence)
            parser.set_defaults(**config)
            args = parser.parse_args()
        except Exception as e:
            parser.error(f"Failed to load config: {e}")
    
    # Validate argument combinations
    mode_count = sum([
        args.update_captions_only,
        args.move_captions is not None,
        args.dry_run
    ])
    
    if mode_count > 1:
        parser.error("Only one operation mode can be specified at a time")
    
    return args


def dhash(image, hash_size=8):
    """Calculate difference hash (dHash) for perceptual image comparison."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute dhash: {e}")
        return 0


def hamming_distance(hash1, hash2):
    """Calculate number of differing bits between two hashes."""
    return bin(hash1 ^ hash2).count('1')


class NudeNetAnalyzer:
    """Turns raw detections into natural human language without spatial phantom people."""
    
    def __init__(self, detections, width, height):
        self.dets = detections
        self.w = width
        self.h = height
        self.counts = {}
        for d in detections:
            c = d['class']
            self.counts[c] = self.counts.get(c, 0) + 1

    def _get_people_count_and_gender(self):
        """Heuristic to guess number of people and gender based on body parts."""
        faces_f = self.counts.get("FACE_FEMALE", 0)
        faces_m = self.counts.get("FACE_MALE", 0)
        
        breasts = self.counts.get("FEMALE_BREAST_EXPOSED", 0) + self.counts.get("FEMALE_BREAST_COVERED", 0)
        buttocks = self.counts.get("BUTTOCKS_EXPOSED", 0) + self.counts.get("BUTTOCKS_COVERED", 0)
        
        genitals_f = self.counts.get("FEMALE_GENITALIA_EXPOSED", 0) + self.counts.get("FEMALE_GENITALIA_COVERED", 0)
        genitals_m = self.counts.get("MALE_GENITALIA_EXPOSED", 0) + self.counts.get("MALE_GENITALIA_COVERED", 0)

        est_people_f = max(faces_f, genitals_f, math.ceil(breasts/2))
        est_people_m = max(faces_m, genitals_m)
        
        if est_people_f == 0 and est_people_m == 0:
            if breasts > 0 or genitals_f > 0: 
                est_people_f = 1
            elif genitals_m > 0: 
                est_people_m = 1
            elif buttocks > 0:
                est_people_f = max(1, math.ceil(buttocks/2))
            else: 
                est_people_f = 1

        total = est_people_f + est_people_m
        
        if est_people_f > 0 and est_people_m == 0:
            gender_str = f"{est_people_f} woman" if est_people_f == 1 else f"{est_people_f} women"
        elif est_people_m > 0 and est_people_f == 0:
            gender_str = f"{est_people_m} man" if est_people_m == 1 else f"{est_people_m} men"
        else:
            gender_str = f"{est_people_f} women and {est_people_m} men"
        
        return total, gender_str

    def _get_nudity_state(self):
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

    def _get_visible_features(self):
        """Collect visible features without spatial grouping to avoid phantom people."""
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

    def generate_smart_caption(self):
        """Generate intelligent caption based on detected features."""
        count, gender_str = self._get_people_count_and_gender()
        state = self._get_nudity_state()
        
        caption = f"This image shows {gender_str} who {'is' if count==1 else 'are'} {state}."

        features = self._get_visible_features()
        if features and len(features) <= 4:
            feat_str = ", ".join(features[:-1]) + (" and " + features[-1] if len(features) > 1 else features[0])
            caption += f" Visible features include {feat_str}."
        elif features:
            caption += f" The image shows {features[0]}, {features[1]} and other exposed areas."
        
        return caption


class CaptionEngine:
    """Florence-2 VLM caption generation engine."""
    
    def __init__(self, device, args=None):
        self.args = args
        self.model = None
        self.processor = None
        self.device = device
        self.dtype = None
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
            self.logger.error(f"Failed to initialize VLM: {e}", exc_info=args.debug if args else False)
            self.model = None

    def describe(self, img):
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
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
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
            self.logger.error(f"Florence-2 generation failed: {e}", exc_info=self.args.debug if self.args else False)
            return ""


class Pipeline:
    """Main processing pipeline for dataset generation."""
    
    def __init__(self, args):
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

        # Initialize detector
        self.logger.info("Loading NudeDetector...")
        try:
            self.detector = NudeDetector()
            self.logger.info("NudeDetector loaded successfully")
        except Exception as e:
            self.logger.critical(f"Failed to load NudeDetector: {e}")
            raise

        # Initialize captioner
        self.captioner = None
        if args.use_captioning:
            try:
                self.captioner = CaptionEngine(args.device, args)
            except Exception as e:
                self.logger.error(f"Failed to initialize caption engine: {e}")

        # State management
        self.counter = 0
        self.processed_files = set()
        self.start_time = None

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
                
                self.logger.info(f"Resumed from checkpoint: {len(self.processed_files)} files already processed")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    def _save_checkpoint(self):
        """Save processing progress to checkpoint file."""
        try:
            checkpoint = {
                'processed_files': list(self.processed_files),
                'counter': self.counter,
                'stats': self.stats,
                'timestamp': time.time()
            }
            
            checkpoint_path = Path(self.args.checkpoint_file)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)
            
            self.logger.debug("Checkpoint saved")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def print_stats(self):
        """Print comprehensive statistics summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "="*70)
        print("📊 PROCESSING STATISTICS")
        print("="*70)
        print(f"✅ Processed frames:        {self.stats['processed']}")
        print(f"⏭️  Skipped (no targets):    {self.stats['skipped_no_targets']}")
        print(f"🔁 Skipped (duplicates):    {self.stats['skipped_duplicate_frames']}")
        print(f"🎯 Total detections:        {self.stats['total_detections']}")
        print(f"⬛ Censored regions:        {self.stats['censored_regions']}")
        print(f"❌ Errors encountered:      {self.stats['errors']}")
        print(f"💾 Images saved:            {self.counter}")
        print(f"⏱️  Total time:              {elapsed:.2f}s")
        if self.counter > 0:
            print(f"⚡ Average per image:       {elapsed/self.counter:.2f}s")
        print("="*70 + "\n")

    def xywh_to_xyxy_safe(self, box):
        """Converts NudeNet [x,y,w,h] to [x1,y1,x2,y2] and clamps to target."""
        x, y, w, h = box
        return [
            max(0, min(int(x), self.target_w)),
            max(0, min(int(y), self.target_h)),
            max(0, min(int(x + w), self.target_w)),
            max(0, min(int(y + h), self.target_h))
        ]

    def apply_scaling(self, box, scale_factor):
        """Scales box from center with edge-aware logic."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        new_w = w * (1.0 + scale_factor)
        new_h = h * (1.0 + scale_factor)
        
        new_x1 = cx - new_w/2
        new_y1 = cy - new_h/2
        new_x2 = cx + new_w/2
        new_y2 = cy + new_h/2
        
        edge_threshold = 5
        
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
        
        # Copy missing captions to censored
        for stem, src_file in tqdm(uncensored_txts.items(), desc="Syncing to censored"):
            if stem not in censored_txts:
                try:
                    shutil.copy2(str(src_file), str(self.censored_dir / src_file.name))
                    copied_to_censored += 1
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Failed to copy {src_file.name}: {e}")
        
        # Copy missing captions to uncensored
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
            # Handle special modes
            if self.args.move_captions:
                self.manage_captions()
                return
            
            if self.args.update_captions_only:
                self.run_caption_update()
                return
            
            if self.args.dry_run:
                self.run_dry_run()
                return
            
            # Normal processing
            self.run_full_processing()
            
        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            self.logger.critical(f"Fatal error during processing: {e}", exc_info=True)
            self._save_checkpoint()
            raise
        finally:
            self.print_stats()
            
            # Cleanup checkpoint if completed successfully
            if not self.args.resume and Path(self.args.checkpoint_file).exists():
                try:
                    Path(self.args.checkpoint_file).unlink()
                except:
                    pass

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
        
        file_types = defaultdict(int)
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
        print(f"  - Box scale: {self.args.box_scale}")
        print(f"  - Output format: {self.args.output_format}")
        print(f"  - Output quality: {self.args.output_quality}")
        print(f"  - VLM captioning: {'Enabled' if self.args.use_captioning else 'Disabled'}")
        print(f"  - Frame similarity: {self.args.frame_similarity_threshold}")

    def run_caption_update(self):
        """Update captions for existing images."""
        self.logger.info("MODE: Updating Captions Only...")
        files = sorted([f for f in self.uncensored_dir.glob("*.jpg")])
        self.logger.info(f"Found {len(files)} existing images")

        errors = []
        
        for f in tqdm(files, desc="Updating captions"):
            try:
                # Load image
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

                # Save caption
                txt_path = f.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(final_caption)
                
                self.logger.debug(f"Saved: {txt_path.name} | Length: {len(final_caption)} chars")
                self.stats['processed'] += 1

            except Exception as e:
                errors.append((f.name, str(e)))
                self.stats['errors'] += 1
                self.logger.error(f"Error updating {f.name}: {e}")
        
        if errors:
            self.logger.warning(f"Failed to update {len(errors)} captions")
            for fname, error in errors[:10]:  # Show first 10
                self.logger.debug(f"  - {fname}: {error}")

    def run_full_processing(self):
        """Process all input files."""
        input_path = Path(self.args.input)
        files = sorted([
            f for f in input_path.rglob("*") 
            if f.is_file() and not f.name.startswith('.')
        ])
        
        self.logger.info(f"Processing {len(files)} files from {input_path}")
        
        # Filter already processed files if resuming
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
                
                # Mark as processed
                self.processed_files.add(str(f))
                
                # Periodic checkpoint saving
                if len(self.processed_files) % 50 == 0:
                    self._save_checkpoint()
                
            except IOError as e:
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
                self.logger.error(f"Unexpected error processing {f.name}: {e}", exc_info=self.args.debug)
        
        # Final checkpoint
        self._save_checkpoint()
        
        if errors:
            self.logger.warning(f"Encountered {len(errors)} errors during processing")
            if self.args.debug:
                for fname, error in errors[:20]:  # Show first 20 in debug mode
                    self.logger.debug(f"  - {fname}: {error}")

    def process_video_intelligent(self, video_path):
        """Intelligently extract unique frames from video using perceptual hashing."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sample_interval = max(1, int(fps * 2))

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

                    if prev_hash is None or hamming_distance(current_hash, prev_hash) > self.args.frame_similarity_threshold:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        self.process_frame(img, f"{video_path.stem}_f{frame_idx:05d}")
                        prev_hash = current_hash
                        extracted += 1
                    else:
                        self.stats['skipped_duplicate_frames'] += 1

                frame_idx += 1
                
                # Periodic memory cleanup for long videos
                if frame_idx % 100 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            cap.release()
            self.logger.debug(f"Extracted {extracted} unique frames from {video_path.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to process video {video_path.name}: {e}", exc_info=self.args.debug)
            raise

    def process_gif_intelligent(self, gif_path):
        """Intelligently extract unique frames from GIF using perceptual hashing."""
        try:
            g = Image.open(gif_path)
            prev_hash = None
            extracted = 0

            for i, frame in enumerate(ImageSequence.Iterator(g)):
                if i > 50:  # Limit GIF frames
                    break

                frame_rgb = frame.convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
                current_hash = dhash(frame_bgr)

                if prev_hash is None or hamming_distance(current_hash, prev_hash) > self.args.frame_similarity_threshold:
                    self.process_frame(frame_rgb, f"{gif_path.stem}_g{i:03d}")
                    prev_hash = current_hash
                    extracted += 1
                else:
                    self.stats['skipped_duplicate_frames'] += 1

            self.logger.debug(f"Extracted {extracted} unique frames from {gif_path.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to process GIF {gif_path.name}: {e}", exc_info=self.args.debug)
            raise

    def process_frame(self, pil_img, source_name):
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
                        'box': self.apply_scaling(xyxy, self.args.box_scale),
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

            # Create censored version
            censored_img = img.copy()
            draw = ImageDraw.Draw(censored_img)
            for t in censor_targets:
                draw.rectangle(t['box'], fill="black")

            # Debug visualization
            if self.args.debug:
                self._create_debug_visualization(img, valid_dets, censor_targets, source_name)

            # Save outputs
            fname = f"{self.counter:06d}"
            self.counter += 1
            self.stats['processed'] += 1

            # Determine file extension
            ext = self.args.output_format
            save_params = {}
            
            if ext == 'jpg':
                save_params = {'quality': self.args.output_quality, 'optimize': True}
            elif ext == 'webp':
                save_params = {'quality': self.args.output_quality, 'method': 6}
            elif ext == 'png':
                save_params = {'optimize': True}

            img.save(self.uncensored_dir / f"{fname}.{ext}", **save_params)
            censored_img.save(self.censored_dir / f"{fname}.{ext}", **save_params)
            
            with open(self.uncensored_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
                f.write(final_caption)

            self.logger.debug(
                f"Saved: {fname}.{ext} | Detections: {len(valid_dets)} | "
                f"Censored: {len(censor_targets)} | Caption: {len(final_caption)} chars"
            )

        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to process frame {source_name}: {e}", exc_info=self.args.debug)
            raise

    def _create_debug_visualization(self, img, valid_dets, censor_targets, source_name):
        """Create debug visualization with detection overlays."""
        try:
            from PIL import ImageFont
            
            debug_img = img.copy()
            
            # Load fonts
            try:
                # Try common font paths
                font_paths = [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "C:\\Windows\\Fonts\\arial.ttf"
                ]
                font_large = font_small = None
                for font_path in font_paths:
                    if Path(font_path).exists():
                        font_large = ImageFont.truetype(font_path, 14)
                        font_small = ImageFont.truetype(font_path, 11)
                        break
                
                if not font_large:
                    font_large = font_small = ImageFont.load_default()
            except:
                font_large = font_small = ImageFont.load_default()
            
            # Create overlay for semi-transparent backgrounds
            overlay = Image.new('RGBA', debug_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            dd = ImageDraw.Draw(debug_img)
            
            # Collect statistics
            class_counts = defaultdict(int)
            class_areas = defaultdict(float)
            class_scores = defaultdict(list)
            total_image_area = self.target_w * self.target_h
            
            # Draw detection boxes
            for idx, d in enumerate(valid_dets):
                is_censored = d['class'] in CENSOR_CLASSES
                color = (255, 0, 0) if is_censored else (0, 200, 255)
                
                dd.rectangle(d['box'], outline=color, width=2)
                
                # Shortened label
                short = (d['class'].replace("_EXPOSED", "").replace("_COVERED", "c")
                        .replace("FEMALE_", "F").replace("MALE_", "M").replace("_", ""))
                label = f"{short} {d['score']:.2f}"
                
                # Semi-transparent label background
                overlay_draw.rectangle(
                    [(d['box'][0], d['box'][1] - 18), (d['box'][0] + len(label) * 7 + 4, d['box'][1])],
                    fill=(0, 0, 0, 180)
                )
                
                # Update statistics
                class_counts[d['class']] += 1
                class_scores[d['class']].append(d['score'])
                box_area = (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1])
                class_areas[d['class']] += box_area
            
            # Composite overlay
            debug_img = Image.alpha_composite(debug_img.convert('RGBA'), overlay).convert('RGB')
            dd = ImageDraw.Draw(debug_img)
            
            # Draw labels
            for idx, d in enumerate(valid_dets):
                is_censored = d['class'] in CENSOR_CLASSES
                color = "red" if is_censored else "cyan"
                short = (d['class'].replace("_EXPOSED", "").replace("_COVERED", "c")
                        .replace("FEMALE_", "F").replace("MALE_", "M").replace("_", ""))
                label = f"{short} {d['score']:.2f}"
                dd.text((d['box'][0] + 2, d['box'][1] - 16), label, fill=color, font=font_small)
            
            # Draw info bar
            bar_height = 55
            bar_y = self.target_h - bar_height
            
            overlay2 = Image.new('RGBA', debug_img.size, (0, 0, 0, 0))
            overlay2_draw = ImageDraw.Draw(overlay2)
            overlay2_draw.rectangle([(0, bar_y), (self.target_w, self.target_h)], fill=(0, 0, 0, 220))
            debug_img = Image.alpha_composite(debug_img.convert('RGBA'), overlay2).convert('RGB')
            dd = ImageDraw.Draw(debug_img)
            
            # Info text
            y = bar_y + 5
            dd.text((10, y), f"SRC: {source_name[:45]}", fill="yellow", font=font_large)
            dd.text((self.target_w - 200, y), f"OUT: #{self.counter:06d}.{self.args.output_format}", 
                   fill="yellow", font=font_large)
            
            y += 18
            det_text = (f"DETS: {len(valid_dets)}  CENSOR: {len(censor_targets)}  "
                       f"THRESH: {self.args.score_threshold}  SCALE: {self.args.box_scale}")
            dd.text((10, y), det_text, fill="white", font=font_small)
            
            y += 16
            parts = []
            for cls in sorted(class_counts.keys()):
                count = class_counts[cls]
                avg_score = sum(class_scores[cls]) / len(class_scores[cls])
                area_pct = class_areas[cls] / total_image_area * 100
                short = (cls.replace("_EXPOSED", "").replace("_COVERED", "c")
                        .replace("FEMALE_", "F").replace("MALE_", "M").replace("_", ""))
                parts.append(f"{short}:{count}x{avg_score:.2f}({area_pct:.1f}%)")
            
            labels_line = " | ".join(parts) if parts else "no detections"
            dd.text((10, y), labels_line[:120], fill="cyan", font=font_small)
            
            # Save debug image
            debug_img.save(self.debug_dir / f"{self.counter:06d}.jpg", quality=85)
            
        except Exception as e:
            self.logger.warning(f"Failed to create debug visualization: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug, args.log_file)
    
    logger.info("="*70)
    logger.info("🚀 Ultimate Intelligent Dataset Generator")
    logger.info("="*70)
    
    # Log configuration
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Target Size: {args.target_size}")
    logger.info(f"Device: {args.device}")
    
    if args.update_captions_only:
        logger.info("Mode: Caption Update Only")
    elif args.move_captions:
        logger.info(f"Mode: Move Captions ({args.move_captions})")
    elif args.dry_run:
        logger.info("Mode: Dry Run")
    elif args.resume:
        logger.info("Mode: Resume from Checkpoint")
    else:
        logger.info("Mode: Full Processing")
    
    try:
        pipeline = Pipeline(args)
        pipeline.process_data()
        logger.info("✅ Processing completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
