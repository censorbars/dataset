# 🎯 Ultimate Intelligent Dataset Generator

Advanced AI-powered dataset creation tool with intelligent frame extraction, perceptual deduplication, and smart captioning for training computer vision models. Automatically processes images, videos, and GIFs while applying censorship and generating natural language descriptions.

## ✨ Key Features

### 🧠 Intelligent Processing

- **Perceptual Hashing**: Uses dHash algorithm for frame-level deduplication, preventing identical frames from overloading your dataset
- **Smart People Detection**: Accurately counts subjects without creating "phantom people" from spatial analysis
- **Adaptive Frame Extraction**: Automatically samples unique frames from videos/GIFs based on visual similarity
- **NudeNet Integration**: High-accuracy body part detection with configurable confidence thresholds

### 📝 Advanced Captioning

- **Two-Tier Captioning System**:
  - **Smart Analysis**: Rule-based natural language generation from NudeNet detections
  - **Florence-2 VLM**: Optional deep learning visual description (transformers required)
- **Human-Readable Labels**: Converts technical classifications into natural descriptions
- **Context-Aware Descriptions**: Analyzes gender, nudity state, and visible features intelligently

### ⚡ Production-Ready Performance

- **Batch Processing**: Handles entire directories recursively
- **Memory Efficient**: Processes videos frame-by-frame without loading entire files
- **Progress Tracking**: Real-time tqdm progress bars with ETA
- **Comprehensive Statistics**: Detailed processing metrics and debug outputs

### 🎨 Flexible Censorship

- **Configurable Black Boxes**: Adjustable size scaling (-0.3 to +0.5)
- **Dual Output**: Generates both censored and uncensored versions
- **Smart Detection**: Only saves frames with censorship targets (clean dataset policy)

## 📋 Requirements

```bash
# Core Dependencies
python >= 3.8
opencv-python >= 4.5.0
torch >= 1.9.0
numpy >= 1.19.0
pillow >= 8.0.0
tqdm >= 4.60.0
nudenet >= 2.0.0

# Optional (for Florence-2 VLM captioning)
transformers >= 4.30.0
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/censorbars/dataset
cd dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python torch numpy pillow tqdm nudenet

# Optional: Install transformers for VLM captioning
pip install transformers
```

## 💻 Usage

### Basic Processing

```bash
python dataset.py \
    --input ./dataset_raw \
    --output ./dataset_processed
```

### Advanced Configuration

```bash
python dataset.py \
    --input ./raw_media \
    --output ./processed_output \
    --target-size 768x1024 \
    --score-threshold 0.35 \
    --box-scale -0.3 \
    --frame-similarity-threshold 8 \
    --use-captioning \
    --debug \
    --device cuda
```

### Caption Update Mode

Regenerate captions for existing images without reprocessing:

```bash
python dataset.py \
    --update-captions-only \
    --use-captioning \
    --output ./dataset_processed
```

## 🎛️ Parameters

| Parameter                      | Default     | Description                                             |
| ------------------------------ | ----------- | ------------------------------------------------------- |
| `--input`                      | `./raw`     | Input folder containing raw images/videos/GIFs          |
| `--output`                     | `./dataset` | Output root directory                                   |
| `--target-size`                | `768x1024`  | Output resolution (WxH)                                 |
| `--score-threshold`            | `0.35`      | Minimum NudeNet confidence (0.0-1.0)                    |
| `--box-scale`                  | `-0.3`      | Censor box resize: -0.3=30% smaller, 0.3=30% larger     |
| `--frame-similarity-threshold` | `8`         | Max hamming distance for deduplication (lower=stricter) |
| `--use-captioning`             | `False`     | Enable Florence-2 VLM descriptions                      |
| `--update-captions-only`       | `False`     | Only regenerate .txt files for existing images          |
| `--debug`                      | `False`     | Enable verbose logging and debug visuals                |
| `--device`                     | `cuda`      | Processing device (cuda/cpu/mps)                        |

## 📁 Output Structure

<table>
<tr>
<td>Censored</td>
<td>Uncensored</td>
<td>Debug</td>
</tr>
<tr>
<td><img src="dataset/dataset_censored/000000.jpg"></td>
<td><img src="dataset/dataset_uncensored/000000.jpg"></td>
<td><img src="dataset/debug_visuals/000000.jpg"></td>
</tr>
</table>

```
dataset/
├── dataset_uncensored/
│   ├── 000000.jpg          # Original processed image
│   ├── 000000.txt          # Caption file
│   ├── 000001.jpg
│   ├── 000001.txt
│   └── ...
├── dataset_censored/
│   ├── 000000.jpg          # Black-box censored version
│   ├── 000001.jpg
│   └── ...
└── debug_visuals/          # Only with --debug flag
    ├── 000000.jpg          # Annotated detection boxes
    └── ...
```

## 📊 Performance Example

Your real-world statistics demonstrate excellent efficiency:

```
============================================================
📊 PROCESSING STATISTICS
============================================================
✅ Processed frames:        1,632
⏭️ Skipped (no targets):    1,661
🔁 Skipped (duplicates):    1,503  ← 47% duplicate reduction!
🎯 Total detections:        8,530
⬛ Censored regions:        3,085
💾 Images saved:            1,632
============================================================
```

**Key Insights**:

- **47% fewer frames** processed thanks to perceptual hashing deduplication
- **Clean dataset**: Only frames with censorship targets included
- **High detection rate**: Average 5.2 detections per saved image
- **Processing efficiency**: 1.9 censored regions per image

## 🎨 Caption Examples

```
This image shows 2 women who are fully nude. Visible features include exposed
breasts, exposed genitalia and exposed buttocks.
```

✅ Accurate count with concise feature description

## 🔧 Troubleshooting

### Out of Memory (Videos)

- Reduce `--frame-similarity-threshold` to extract fewer frames
- Lower `--target-size` resolution
- Process videos separately with smaller batches

### Too Many Duplicate Frames

- Increase `--frame-similarity-threshold` (try 10-15 for very similar content)

### Missing Detections

- Lower `--score-threshold` to 0.25-0.30
- Check input image quality and resolution

### Florence-2 Not Loading

```bash
pip install --upgrade transformers accelerate
# Ensure you have 8GB+ GPU VRAM for Florence-2-large
```

## 🛡️ Ethical Usage

This tool is designed for:

- ✅ Research dataset creation
- ✅ Content moderation system training
- ✅ Age-restricted content filtering
- ✅ Computer vision model development

**Responsible AI**: Always obtain proper permissions and comply with local regulations when processing sensitive content.
