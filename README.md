# 🎯 Ultimate Intelligent Dataset Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/censorbars/dataset/graphs/commit-activity)

Advanced AI-powered dataset creation tool with intelligent frame extraction, perceptual deduplication, and smart captioning for training computer vision models. Automatically processes images, videos, and GIFs while applying censorship and generating natural language descriptions.

![Demo](scr.png)

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Complete Parameters Reference](#-complete-parameters-reference)
- [Configuration Files](#-configuration-files)
- [Output Structure](#-output-structure)
- [Performance](#-performance-example)
- [Caption Management](#-caption-management)
- [Caption Examples](#-caption-examples)
- [Troubleshooting](#-troubleshooting)
- [Ethical Usage](#-ethical-usage)

---

## ✨ Key Features

### 🧠 Intelligent Processing

- **Perceptual Hashing**: Uses dHash algorithm for frame-level deduplication, preventing identical frames from overloading your dataset
- **Smart People Detection**: Accurately counts subjects without creating "phantom people" from spatial analysis
- **Adaptive Frame Extraction**: Automatically samples unique frames from videos/GIFs based on visual similarity
- **NudeNet Integration**: High-accuracy body part detection with configurable confidence thresholds
- **Resume Capability**: Checkpoint system allows resuming interrupted processing jobs

### 📝 Advanced Captioning

- **Two-Tier Captioning System**:
  - **Smart Analysis**: Rule-based natural language generation from NudeNet detections
  - **Florence-2 VLM**: Optional deep learning visual description (transformers required)
- **Human-Readable Labels**: Converts technical classifications into natural descriptions
- **Context-Aware Descriptions**: Analyzes gender, nudity state, and visible features intelligently
- **Caption Management**: Move, sync, or update captions across datasets

### ⚡ Production-Ready Performance

- **Batch Processing**: Handles entire directories recursively
- **Memory Efficient**: Processes videos frame-by-frame without loading entire files
- **Progress Tracking**: Real-time tqdm progress bars with ETA
- **Comprehensive Logging**: Professional logging system with file and console output
- **Error Recovery**: Graceful error handling with detailed diagnostics
- **Performance Metrics**: Detailed processing statistics and timing information

### 🎨 Flexible Output

- **Configurable Censorship**: Adjustable black box sizing (-0.3 to +0.5)
- **Multiple Formats**: JPG, PNG, or WebP output with quality control
- **Dual Output**: Generates both censored and uncensored versions
- **Smart Detection**: Only saves frames with censorship targets (clean dataset policy)
- **Debug Visualizations**: Optional annotated images with detection overlays

---

## 🔧 Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows 10/11
- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 10GB+ for processing large datasets
- **GPU** (optional but recommended):
  - CUDA-capable GPU with 6GB+ VRAM for Florence-2
  - 12GB+ VRAM recommended for large batch processing

### Software Dependencies

```bash
# Core (Required)
python >= 3.8
opencv-python >= 4.5.0
torch >= 1.9.0
numpy >= 1.19.0
pillow >= 8.0.0
tqdm >= 4.60.0
nudenet >= 2.0.0

# Optional (for Florence-2 VLM captioning)
transformers >= 4.30.0

# Optional (for YAML config files)
pyyaml >= 5.4.0
```

---

## 🚀 Installation

### Option 1: pip (Recommended)

```bash
# Clone repository
git clone https://github.com/censorbars/dataset.git
cd dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Optional: Install Florence-2 support
pip install transformers accelerate

# Optional: Install YAML support
pip install pyyaml
```

### Option 2: Conda

```bash
conda create -n dataset python=3.10
conda activate dataset
pip install -r requirements.txt
```

### Option 3: Google Colab

```python
# Install in Colab
!git clone https://github.com/censorbars/dataset.git
%cd dataset
!pip install -r requirements.txt
```

**✅ Verified compatible with Google Colab T4 GPUs**

---

## 🏃 Quick Start

### Basic Usage

```bash
# Basic processing
python dataset.py --input ./raw --output ./dataset

# With Florence-2 captioning
python dataset.py \
    --input ./raw \
    --output ./dataset \
    --use-captioning \
    --debug
```

### Using Config Files

```bash
# Create config.json
cat > config.json << EOF
{
  "input": "./raw_media",
  "output": "./dataset",
  "target_size": "1024x1536",
  "score_threshold": 0.40,
  "use_captioning": true,
  "device": "cuda"
}
EOF

# Run with config
python dataset.py --config config.json
```

---

## 💻 Usage Examples

### Example 1: High-Quality Dataset Generation

```bash
python dataset.py \
    --input ./raw_media \
    --output ./hq_dataset \
    --target-size 1024x1536 \
    --score-threshold 0.40 \
    --output-format webp \
    --output-quality 95 \
    --use-captioning \
    --device cuda
```

### Example 2: Resume Interrupted Processing

```bash
# Start processing
python dataset.py \
    --input ./large_dataset \
    --output ./output \
    --resume

# If interrupted, resume from checkpoint
python dataset.py \
    --input ./large_dataset \
    --output ./output \
    --resume
```

### Example 3: Video Processing with Strict Deduplication

```bash
python dataset.py \
    --input ./videos \
    --output ./video_dataset \
    --frame-similarity-threshold 5 \
    --box-scale 0 \
    --debug
```

### Example 4: Caption-Only Update

Regenerate captions without reprocessing images:

```bash
python dataset.py \
    --output ./existing_dataset \
    --update-captions-only \
    --use-captioning \
    --trigger-word "<concept>"
```

### Example 5: Dry Run Preview

Preview what would be processed without modifying files:

```bash
python dataset.py \
    --input ./raw \
    --output ./dataset \
    --dry-run
```

**Output:**

```
🔍 DRY RUN MODE - No files will be modified
📁 Input folder: ./raw
📁 Output folder: ./dataset
📊 Found 1,234 files

File breakdown:
  - Images: 850
  - Videos: 234
  - Gifs: 150

⚙️  Settings:
  - Target size: 768x1024
  - Score threshold: 0.35
  - Output format: jpg
  - VLM captioning: Enabled
```

### Example 6: Google Colab Processing

```python
!python dataset.py \
    --input "/content/drive/MyDrive/raw" \
    --output "/content/drive/MyDrive/dataset" \
    --use-captioning \
    --device cuda \
    --resume \
    --debug
```

---

## 🎛️ Complete Parameters Reference

### Input/Output

| Parameter  | Type   | Default     | Description                                    |
| ---------- | ------ | ----------- | ---------------------------------------------- |
| `--input`  | string | `./raw`     | Input folder containing raw images/videos/GIFs |
| `--output` | string | `./dataset` | Output root directory                          |
| `--config` | string | -           | Path to JSON/YAML config file                  |

### Operation Modes

| Parameter                | Type   | Default | Description                                                |
| ------------------------ | ------ | ------- | ---------------------------------------------------------- |
| `--update-captions-only` | flag   | `False` | Only regenerate .txt files for existing images             |
| `--move-captions`        | choice | -       | Move captions: `to-censored`, `to-uncensored`, `sync-both` |
| `--dry-run`              | flag   | `False` | Preview processing without modifying files                 |
| `--resume`               | flag   | `False` | Resume from last checkpoint                                |

### Processing Parameters

| Parameter                      | Type   | Default    | Description                                             |
| ------------------------------ | ------ | ---------- | ------------------------------------------------------- |
| `--target-size`                | string | `768x1024` | Output resolution in WxH format                         |
| `--score-threshold`            | float  | `0.35`     | Minimum NudeNet confidence (0.0-1.0)                    |
| `--box-scale`                  | float  | `-0.3`     | Censor box resize: -0.3=30% smaller, 0.3=30% larger     |
| `--frame-similarity-threshold` | int    | `8`        | Max hamming distance for deduplication (lower=stricter) |
| `--batch-size`                 | int    | `1`        | Frames to process in parallel (experimental)            |

### Output Settings

| Parameter          | Type   | Default | Description                         |
| ------------------ | ------ | ------- | ----------------------------------- |
| `--output-format`  | choice | `jpg`   | Output format: `jpg`, `png`, `webp` |
| `--output-quality` | int    | `95`    | Quality for jpg/webp (1-100)        |

### Captioning

| Parameter          | Type   | Default     | Description                                                |
| ------------------ | ------ | ----------- | ---------------------------------------------------------- |
| `--use-captioning` | flag   | `False`     | Enable Florence-2 VLM descriptions                         |
| `--trigger-word`   | string | `[trigger]` | Prefix token for training (e.g., `[trigger]`, `<concept>`) |

### Checkpointing

| Parameter           | Type   | Default                       | Description             |
| ------------------- | ------ | ----------------------------- | ----------------------- |
| `--checkpoint-file` | string | `.processing_checkpoint.json` | Path to checkpoint file |

### System

| Parameter    | Type   | Default          | Description                                |
| ------------ | ------ | ---------------- | ------------------------------------------ |
| `--device`   | string | `cuda`           | Device for inference: `cuda`, `cpu`, `mps` |
| `--debug`    | flag   | `False`          | Enable verbose logs and debug visuals      |
| `--log-file` | string | `processing.log` | Path to log file                           |

---

## 📝 Configuration Files

### JSON Configuration

```json
{
  "input": "./raw_media",
  "output": "./dataset",
  "target_size": "1024x1536",
  "score_threshold": 0.4,
  "box_scale": -0.2,
  "frame_similarity_threshold": 8,
  "output_format": "webp",
  "output_quality": 90,
  "use_captioning": true,
  "trigger_word": "<concept>",
  "device": "cuda",
  "resume": true,
  "debug": false
}
```

### YAML Configuration

```yaml
# dataset_config.yaml
input: ./raw_media
output: ./dataset
target_size: 1024x1536
score_threshold: 0.40
box_scale: -0.2
frame_similarity_threshold: 8
output_format: webp
output_quality: 90
use_captioning: true
trigger_word: "<concept>"
device: cuda
resume: true
debug: false
```

**Usage:**

```bash
# JSON config
python dataset.py --config config.json

# YAML config (requires pyyaml)
python dataset.py --config config.yaml

# Override config values via CLI
python dataset.py --config config.json --debug --score-threshold 0.5
```

**Note**: Command-line arguments take precedence over config file values.

---

## 📁 Output Structure

```
dataset/
├── dataset_uncensored/
│   ├── 000000.jpg          # Original processed image
│   ├── 000000.txt          # Caption file
│   ├── 000001.webp         # Supports multiple formats
│   ├── 000001.txt
│   └── ...
├── dataset_censored/
│   ├── 000000.jpg          # Black-box censored version
│   ├── 000001.webp
│   └── ...
├── debug_visuals/          # Only with --debug flag
│   ├── 000000.jpg          # Annotated detection boxes
│   └── ...
├── processing.log          # Detailed processing log
└── .processing_checkpoint.json  # Resume checkpoint (auto-cleanup on completion)
```

### Visual Examples

<table>
<tr>
<td><b>Censored</b></td>
<td><b>Uncensored</b></td>
<td><b>Debug</b></td>
</tr>
<tr>
<td><img src="dataset/dataset_censored/000000.jpg" width="250"></td>
<td><img src="dataset/dataset_uncensored/000000.jpg" width="250"></td>
<td><img src="dataset/debug_visuals/000000.jpg" width="250"></td>
</tr>
</table>

---

## 📊 Performance Example

Real-world statistics from production use:

```
======================================================================
📊 PROCESSING STATISTICS
======================================================================
✅ Processed frames:        1,632
⏭️  Skipped (no targets):    1,661
🔁 Skipped (duplicates):    1,503  ← 47% duplicate reduction!
🎯 Total detections:        8,530
⬛ Censored regions:        3,085
❌ Errors encountered:      0
💾 Images saved:            1,632
⏱️  Total time:              2,847.3s
⚡ Average per image:        1.74s
======================================================================
```

**Key Insights**:

- **47% fewer frames** processed thanks to perceptual hashing
- **Clean dataset**: Only frames with censorship targets included
- **High detection rate**: Average 5.2 detections per saved image
- **Efficiency**: 1.9 censored regions per image
- **Speed**: 1.74s per image with Florence-2 on T4 GPU

---

## 🔄 Caption Management

### Move Captions Between Datasets

```bash
# Move captions from uncensored to censored
python dataset.py \
    --output ./dataset \
    --move-captions to-censored

# Move captions from censored to uncensored
python dataset.py \
    --output ./dataset \
    --move-captions to-uncensored

# Sync captions in both directions (copies missing captions)
python dataset.py \
    --output ./dataset \
    --move-captions sync-both
```

### Update Existing Captions

```bash
# Regenerate all captions with new trigger word
python dataset.py \
    --output ./dataset \
    --update-captions-only \
    --trigger-word "<new_concept>" \
    --use-captioning
```

### Workflow Example

```bash
# 1. Initial processing
python dataset.py --input ./raw --output ./dataset

# 2. Review uncensored images
# ... manual review ...

# 3. Update captions with VLM
python dataset.py \
    --output ./dataset \
    --update-captions-only \
    --use-captioning

# 4. Sync captions to censored folder
python dataset.py \
    --output ./dataset \
    --move-captions sync-both
```

---

## 🎨 Caption Examples

### Smart NudeNet Captions

```
[trigger] This image shows 1 woman who is topless.
Visible features include exposed breasts.
```

```
[trigger] This image shows 2 women who are fully nude.
Visible features include exposed breasts, exposed genitalia and other exposed areas.
```

### Florence-2 VLM Captions

```
[trigger] The image is a close-up of a person's lower body,
specifically their breasts. The person is lying on their stomach,
with their legs spread apart and their hands resting on their knees.
The background is blurred, but it appears to be a bed or a couch
with a white blanket. The focus of the image is on the person's
breasts, which are large and prominent. The image is taken
from a slightly elevated angle, looking down on their body.
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

**Problem**: CUDA out of memory when processing videos

**Solution**:

```bash
# Reduce frame extraction rate
python dataset.py --frame-similarity-threshold 5

# Lower output resolution
python dataset.py --target-size 512x768

# Use CPU instead of GPU
python dataset.py --device cpu

# Process in batches (save checkpoint every 50 files)
python dataset.py --resume
```

### Resume Not Working

**Problem**: `--resume` flag doesn't seem to work

**Solution**:

```bash
# Ensure checkpoint file exists
ls -la .processing_checkpoint.json

# Specify custom checkpoint path
python dataset.py --resume --checkpoint-file ./my_checkpoint.json

# Check log file for detailed errors
tail -f processing.log
```

### Too Many Duplicate Frames

**Problem**: Video has very similar frames

**Solution**:

```bash
# Increase threshold for more aggressive deduplication
python dataset.py --frame-similarity-threshold 12

# Enable debug mode to see hamming distances
python dataset.py --debug --frame-similarity-threshold 12
```

### Missing Detections

**Problem**: NudeNet not detecting obvious targets

**Solution**:

```bash
# Lower confidence threshold
python dataset.py --score-threshold 0.25

# Enable debug mode to visualize detections
python dataset.py --score-threshold 0.25 --debug
```

### Config File Errors

**Problem**: Config file not loading

**Solution**:

```bash
# Validate JSON syntax
cat config.json | python -m json.tool

# Check YAML syntax (requires pyyaml)
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Use absolute paths in config
{
  "input": "/full/path/to/raw",
  "output": "/full/path/to/dataset"
}
```

### Florence-2 Errors

**Problem**: `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`

**Solution**: ✅ **Fixed!** This was a dtype mismatch issue resolved by:

- Converting `pixel_values` to model dtype (float16 on CUDA)
- Adding `use_cache=False` to generation parameters

**Problem**: Florence-2 not loading or crashes

**Solution**:

```bash
# Update dependencies
pip install --upgrade transformers torch

# Verify GPU has 8GB+ VRAM
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9)"

# Try CPU mode if GPU insufficient
python dataset.py --use-captioning --device cpu
```

### Log File Issues

**Problem**: Log file not being created

**Solution**:

```bash
# Check write permissions
touch processing.log && echo "test" > processing.log

# Specify custom log path
python dataset.py --log-file /path/to/custom.log

# Check logs in console (always outputs to console)
python dataset.py --debug
```

---

## 🛡️ Ethical Usage

This tool is designed for:

- ✅ Research dataset creation
- ✅ Content moderation system training
- ✅ Age-restricted content filtering
- ✅ Computer vision model development
- ✅ Academic research with proper IRB approval

### Responsible AI Guidelines

⚠️ **Important**: Always obtain proper permissions and comply with local regulations when processing sensitive content.

- Respect privacy and consent
- Follow platform terms of service
- Comply with GDPR/CCPA where applicable
- Do not use for harassment or non-consensual content
- Implement appropriate access controls
- Maintain data security and encryption
- Document data sources and usage
- Provide opt-out mechanisms where required

---

## 🙏 Acknowledgments

- [NudeNet](https://github.com/notAI-tech/NudeDetector) for detection models
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) by Microsoft for VLM captioning
- [OpenCV](https://opencv.org/) for video processing
- [Transformers](https://huggingface.co/docs/transformers) by Hugging Face

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/censorbars/dataset/issues)
- **Discussions**: [GitHub Discussions](https://github.com/censorbars/dataset/discussions)

---

**Made with ❤️ for the AI/ML community**
