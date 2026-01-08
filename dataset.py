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


def parse_args():
    parser = argparse.ArgumentParser(description="Ultimate Intelligent Dataset Generator")
    
    parser.add_argument("--input", type=str, default="./raw", help="Input raw folder")
    parser.add_argument("--output", type=str, default="./dataset", help="Output root folder")
    parser.add_argument("--update-captions-only", action="store_true", 
                        help="ONLY regenerate .txt files for existing uncensored images using Smart Logic + VLM.")
    parser.add_argument("--target-size", type=str, default="768x1024", help="WxH (e.g., 768x1024)")
    parser.add_argument("--score-threshold", type=float, default=0.35, help="Minimum confidence score")
    parser.add_argument("--box-scale", type=float, default=-0.3, 
                        help="Resize black box. -0.3 = 30%% smaller. 0.3 = 30%% larger.")
    parser.add_argument("--use-captioning", action="store_true", help="Enable Florence-2 VLM description")
    parser.add_argument("--trigger-word", type=str, default="[trigger]", 
                    help="Trigger token for AI toolkit training (placed at caption start)")
    parser.add_argument("--frame-similarity-threshold", type=int, default=8, 
                        help="Max hamming distance for frame deduplication (lower = more strict)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs and visual debug images")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    
    return parser.parse_args()


def dhash(image, hash_size=8):
    """Calculate difference hash (dHash) for perceptual image comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


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
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not found. Captioning disabled.")
            return
        
        print(f"⏳ Loading Florence-2 on {device}...")
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
            print("✅ Florence-2 Loaded.")
        except Exception as e:
            print(f"❌ VLM Init Failed: {e}")
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
            
            if self.args and self.args.debug:
                print(f"[Florence-2] Generated {len(text)} chars: {text[:100]}...")
            
            return text
            
        except Exception as e:
            print(f"[Florence-2 ERROR] {e}")
            if self.args and self.args.debug:
                import traceback
                traceback.print_exc()
            return ""


class Pipeline:
    """Main processing pipeline for dataset generation."""
    
    def __init__(self, args):
        self.args = args
        self.target_w, self.target_h = map(int, args.target_size.lower().split('x'))

        self.uncensored_dir = Path(args.output) / "dataset_uncensored"
        self.censored_dir = Path(args.output) / "dataset_censored"
        self.debug_dir = Path(args.output) / "debug_visuals"

        for d in [self.uncensored_dir, self.censored_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if args.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        print("⏳ Loading NudeDetector...")
        self.detector = NudeDetector()

        self.captioner = None
        if args.use_captioning:
            self.captioner = CaptionEngine(args.device, args)

        self.counter = 0

        self.stats = {
            'processed': 0,
            'skipped_no_targets': 0,
            'skipped_duplicate_frames': 0,
            'total_detections': 0,
            'censored_regions': 0
        }

    def log(self, msg, level="DEBUG"):
        if self.args.debug:
            prefix = "🔍" if level == "DEBUG" else "📊" if level == "STAT" else "ℹ️"
            tqdm.write(f"{prefix} {msg}")

    def print_stats(self):
        """Print compact statistics summary."""
        if self.args.debug:
            print("\n" + "="*60)
            print("📊 PROCESSING STATISTICS")
            print("="*60)
            print(f"✅ Processed frames:        {self.stats['processed']}")
            print(f"⏭️  Skipped (no targets):    {self.stats['skipped_no_targets']}")
            print(f"🔁 Skipped (duplicates):    {self.stats['skipped_duplicate_frames']}")
            print(f"🎯 Total detections:        {self.stats['total_detections']}")
            print(f"⬛ Censored regions:        {self.stats['censored_regions']}")
            print(f"💾 Images saved:            {self.counter}")
            print("="*60 + "\n")

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

    def process_data(self):
        if self.args.update_captions_only:
            self.run_caption_update()
            return

        self.run_full_processing()
        self.print_stats()

    def run_caption_update(self):
        """Update captions for existing images."""
        print("🔄 MODE: Updating Captions Only...")
        files = sorted([f for f in self.uncensored_dir.glob("*.jpg")])
        print(f"Found {len(files)} existing images.")

        for f in tqdm(files, desc="Updating captions"):
            try:
                img = Image.open(f).convert("RGB")

                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                detections = self.detector.detect(img_bgr)

                valid_dets = []
                for d in detections:
                    if d['score'] >= self.args.score_threshold:
                        d['box'] = self.xywh_to_xyxy_safe(d['box']) 
                        valid_dets.append(d)

                analyzer = NudeNetAnalyzer(valid_dets, img.width, img.height)
                smart_cap = analyzer.generate_smart_caption()

                vlm_cap = ""
                if self.captioner:
                    self.log(f"Running Florence-2 for {f.name}...", "DEBUG")
                    vlm_cap = self.captioner.describe(img)
                    self.log(f"Florence-2 output: {vlm_cap[:100]}...", "DEBUG")

                if vlm_cap:
                    final_caption = f"{self.args.trigger_word} {vlm_cap}"
                    self.log(f"Using VLM caption for {f.name}", "DEBUG")
                else:
                    final_caption = f"{self.args.trigger_word} {smart_cap}"
                    self.log(f"Using Smart caption for {f.name}", "DEBUG")

                txt_path = f.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(final_caption)
                
                self.log(f"Saved: {txt_path.name} | Length: {len(final_caption)} chars", "DEBUG")

            except Exception as e:
                print(f"❌ Error updating {f.name}: {e}")

    def run_full_processing(self):
        """Process all input files."""
        input_path = Path(self.args.input)
        files = sorted([f for f in input_path.rglob("*") if f.is_file() and not f.name.startswith('.')])
        print(f"🚀 Processing {len(files)} raw files...")

        for f in tqdm(files, desc="Processing files"):
            try:
                ext = f.suffix.lower()
                stem = f.stem
                if ext in {'.jpg', '.jpeg', '.png', '.webp'}:
                    i = Image.open(f).convert("RGB")
                    i = ImageOps.exif_transpose(i)
                    self.process_frame(i, stem)
                elif ext in {'.mp4', '.avi', '.mov', '.webm'}:
                    self.process_video_intelligent(f)
                elif ext == ".gif":
                    self.process_gif_intelligent(f)
            except Exception as e:
                self.log(f"Error {f.name}: {e}", "ERROR")

    def process_video_intelligent(self, video_path):
        """Intelligently extract unique frames from video using perceptual hashing."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_interval = max(1, int(fps * 2))

        prev_hash = None
        frame_idx = 0
        extracted = 0

        self.log(f"Video: {video_path.name} | FPS: {fps:.1f} | Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % sample_interval == 0:
                current_hash = dhash(frame)

                if prev_hash is None or hamming_distance(current_hash, prev_hash) > self.args.frame_similarity_threshold:
                    i = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.process_frame(i, f"{video_path.stem}_f{frame_idx:05d}")
                    prev_hash = current_hash
                    extracted += 1
                else:
                    self.stats['skipped_duplicate_frames'] += 1

            frame_idx += 1

        cap.release()
        self.log(f"Extracted {extracted} unique frames from {video_path.name}", "STAT")

    def process_gif_intelligent(self, gif_path):
        """Intelligently extract unique frames from GIF using perceptual hashing."""
        g = Image.open(gif_path)
        prev_hash = None
        extracted = 0

        for i, frame in enumerate(ImageSequence.Iterator(g)):
            if i > 50: break

            frame_rgb = frame.convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
            current_hash = dhash(frame_bgr)

            if prev_hash is None or hamming_distance(current_hash, prev_hash) > self.args.frame_similarity_threshold:
                self.process_frame(frame_rgb, f"{gif_path.stem}_g{i:03d}")
                prev_hash = current_hash
                extracted += 1
            else:
                self.stats['skipped_duplicate_frames'] += 1

        self.log(f"Extracted {extracted} unique frames from {gif_path.name}", "STAT")

    def process_frame(self, pil_img, source_name):
        """Process a single frame/image."""
        img = ImageOps.fit(pil_img, (self.target_w, self.target_h), method=Image.LANCZOS, centering=(0.5, 0.5))

        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        detections = self.detector.detect(img_bgr)

        valid_dets = []
        censor_targets = []

        for d in detections:
            if d['score'] < self.args.score_threshold: continue

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

        if not censor_targets:
            self.stats['skipped_no_targets'] += 1
            return

        analyzer = NudeNetAnalyzer(valid_dets, self.target_w, self.target_h)
        smart_caption = analyzer.generate_smart_caption()

        vlm_text = self.captioner.describe(img) if self.captioner else ""

        if vlm_text:
            final_caption = f"{self.args.trigger_word} {vlm_text}"
        else:
            final_caption = f"{self.args.trigger_word} {smart_caption}"

        censored_img = img.copy()
        draw = ImageDraw.Draw(censored_img)
        for t in censor_targets:
            draw.rectangle(t['box'], fill="black")

        if self.args.debug:
            self._create_debug_visualization(img, valid_dets, censor_targets, source_name)

        fname = f"{self.counter:06d}"
        self.counter += 1
        self.stats['processed'] += 1

        img.save(self.uncensored_dir / f"{fname}.jpg", quality=95)
        censored_img.save(self.censored_dir / f"{fname}.jpg", quality=95)
        with open(self.uncensored_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
            f.write(final_caption)

    def _create_debug_visualization(self, img, valid_dets, censor_targets, source_name):
        """Create debug visualization with detection overlays."""
        from PIL import ImageFont
        
        debug_img = img.copy()
        
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        overlay = Image.new('RGBA', debug_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        dd = ImageDraw.Draw(debug_img)
        
        class_counts = defaultdict(int)
        class_areas = defaultdict(float)
        class_scores = defaultdict(list)
        total_image_area = self.target_w * self.target_h
        
        for idx, d in enumerate(valid_dets):
            is_censored = d['class'] in CENSOR_CLASSES
            color = (255, 0, 0) if is_censored else (0, 200, 255)
            
            dd.rectangle(d['box'], outline=color, width=2)
            
            short = d['class'].replace("_EXPOSED", "").replace("_COVERED", "c").replace("FEMALE_", "F").replace("MALE_", "M").replace("_", "")
            label = f"{short} {d['score']:.2f}"
            
            overlay_draw.rectangle(
                [(d['box'][0], d['box'][1] - 18), (d['box'][0] + len(label) * 7 + 4, d['box'][1])],
                fill=(0, 0, 0, 180)
            )
            
            class_counts[d['class']] += 1
            class_scores[d['class']].append(d['score'])
            box_area = (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1])
            class_areas[d['class']] += box_area
        
        debug_img = Image.alpha_composite(debug_img.convert('RGBA'), overlay).convert('RGB')
        dd = ImageDraw.Draw(debug_img)
        
        for idx, d in enumerate(valid_dets):
            is_censored = d['class'] in CENSOR_CLASSES
            color = "red" if is_censored else "cyan"
            short = d['class'].replace("_EXPOSED", "").replace("_COVERED", "c").replace("FEMALE_", "F").replace("MALE_", "M").replace("_", "")
            label = f"{short} {d['score']:.2f}"
            dd.text((d['box'][0] + 2, d['box'][1] - 16), label, fill=color, font=font_small)
        
        bar_height = 55
        bar_y = self.target_h - bar_height
        
        overlay2 = Image.new('RGBA', debug_img.size, (0, 0, 0, 0))
        overlay2_draw = ImageDraw.Draw(overlay2)
        overlay2_draw.rectangle([(0, bar_y), (self.target_w, self.target_h)], fill=(0, 0, 0, 220))
        debug_img = Image.alpha_composite(debug_img.convert('RGBA'), overlay2).convert('RGB')
        dd = ImageDraw.Draw(debug_img)
        
        y = bar_y + 5
        dd.text((10, y), f"SRC: {source_name[:45]}", fill="yellow", font=font_large)
        dd.text((self.target_w - 200, y), f"OUT: #{self.counter:06d}.jpg", fill="yellow", font=font_large)
        
        y += 18
        det_text = f"DETS: {len(valid_dets)}  CENSOR: {len(censor_targets)}  THRESH: {self.args.score_threshold}  SCALE: {self.args.box_scale}"
        dd.text((10, y), det_text, fill="white", font=font_small)
        
        y += 16
        parts = []
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            avg_score = sum(class_scores[cls]) / len(class_scores[cls])
            area_pct = class_areas[cls] / total_image_area * 100
            short = cls.replace("_EXPOSED", "").replace("_COVERED", "c").replace("FEMALE_", "F").replace("MALE_", "M").replace("_", "")
            parts.append(f"{short}:{count}x{avg_score:.2f}({area_pct:.1f}%)")
        
        labels_line = " | ".join(parts) if parts else "no detections"
        dd.text((10, y), labels_line[:120], fill="cyan", font=font_small)
        
        debug_img.save(self.debug_dir / f"{self.counter:06d}.jpg", quality=85)
        
        log_parts = [f"{cls.split('_')[0]}:{class_counts[cls]}[{sum(class_scores[cls])/len(class_scores[cls]):.2f}]" for cls in sorted(class_counts.keys())]
        self.log(f"#{self.counter:06d} {source_name:30s} | {' '.join(log_parts)}")


if __name__ == "__main__":
    args = parse_args()
    Pipeline(args).process_data()
