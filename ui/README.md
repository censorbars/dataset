# Dataset Creator

<img src="creator.png">

A high-performance, web interface for creating image datasets with target/control pairs. Features intelligent image conversion, dataset structure validation, and a smooth, professional UI/UX optimized for creating thousands of training pairs.

## 🌟 Key Features

### 🎯 Dataset Structure Validation

- **Enforced Consistency**: Once a dataset has a structure (e.g., 1 target + 3 controls), all future additions must match
- **Smart Validation**: Server-side validation prevents structural mismatches
- **Clear Feedback**: Informative error messages guide users to correct selections
- **Visual Indicators**: Dataset buttons show structure (e.g., "7" with "1+3" below)

### 🖼️ Intelligent Image Processing

- **Smart Scaling**: Automatically scales images to cover target dimensions
- **Center Cropping**: Intelligently crops from center to maintain focus
- **Aspect Ratio Correction**: Ensures all dataset images have identical dimensions
- **PNG→JPG Conversion**: High-quality conversion with proper encoding (fixes macOS visibility)
- **Quality Preservation**: 95% JPEG quality with 4:4:4 chroma subsampling

### 🎨 Perfect UI/UX

- **No Layout Shifting**: Placeholder loading prevents jumps and jitters
- **Scroll Position Preserved**: Page stays in place after saving
- **Fixed Modal Height**: Comparison window doesn't resize when switching modes
- **Shimmer Loading**: Elegant placeholder animations
- **Color-Coded Scaling**: Gray for downscale, red for upscale indicators
- **Smooth Transitions**: Professional, quiet interface updates

### ⚡ Performance Optimizations

- **Batch Processing**: Handles 1,500+ images without freezing
- **Progressive Loading**: Shows loading progress with image count
- **Image Caching**: Converted images cached for instant access
- **Debounced Rendering**: Smooth UI updates without lag
- **Lazy Loading**: Images load as needed
- **Document Fragments**: Efficient DOM manipulation

### 🎛️ Flexible Settings

- **10 Resolution Presets**: 256×256 to 1536×1536
- **Custom Dimensions**: Support for any size from 64×4096
- **Live Preview**: See how images will look before saving
- **Easy Access**: Settings panel with gear icon in header
- **Multiple Close Options**: X button or click outside

### 🔄 Professional Workflow

1. Select target image (blue outline)
2. Select control image(s) (green outline)
3. Optional: Press `C` to compare
4. Press `0-9` to save to dataset
5. Modal closes, selection clears, scroll position maintained
6. Continue working immediately

## 📋 Requirements

- **Node.js**: 14+ (recommended 18+)
- **Browser**: Modern browser (Chrome, Firefox, Safari, Edge)
- **RAM**: 4GB+ recommended for large image sets
- **Disk Space**: Varies by dataset size

## 🚀 Quick Start

```bash
# Clone or download the project
cd dataset-creator

# Install dependencies
npm install

# Start the server
npm start

# Open browser
http://localhost:3000
```

## 📦 Installation

### 1. Install Dependencies

```bash
npm install
```

This installs:

- **express** (5.2.1) - Web server
- **multer** (2.0.2) - File upload handling
- **sharp** (0.33.5) - High-performance image processing
- **axios** (1.13.2) - HTTP client
- **cors** (2.8.5) - Cross-origin resource sharing

### 2. Start Server

**Production mode:**

```bash
npm start
```

**Development mode** (auto-restart on changes):

```bash
npm run dev
```

### 3. Access Interface

Open your browser to: `http://localhost:3000`

## 📖 Complete Usage Guide

### Loading Images

#### Option 1: Image URLs

1. Create a `.txt` file with one URL per line:
   ```
   https://example.com/image1.jpg
   https://example.com/image2.jpg
   https://example.com/image3.jpg
   ```
2. Click **"📄 Load Image URLs"**
3. Select your `.txt` file
4. Wait for loading progress

#### Option 2: Local Folder

1. Click **"📁 Load Image Folder"**
2. Select folder containing images
3. All images automatically loaded with dimensions
4. Loading progress shows processing status

### Configuring Image Size

1. Click **gear icon** in header (top-right)
2. Select a preset or use custom dimensions:

**Available Presets:**

- 256×256 (Small, fast processing)
- 512×512 (Standard)
- 768×768 (Medium quality)
- **1024×1024** (Default, recommended)
- 768×1024 (Portrait)
- 1024×768 (Landscape)
- 768×1280 (Tall portrait)
- 1280×720 (HD landscape)
- 1280×1280 (High quality)
- 1536×1536 (Very high quality)

3. Close panel by:
   - Clicking **X** button
   - Clicking **outside** panel
   - Clicking gear icon again

### Creating Your First Dataset

#### Basic Pair (1 Target + 1 Control)

```
1. Click on image A (becomes Target - blue outline)
2. Click on image B (becomes Control #1 - green outline)
3. Press 5 (saves to dataset 5)
4. ✓ Saved! Selection cleared, ready for next pair
```

**Result:**

```
datasets/5/
  ├── target/000000.jpg
  └── control_1/000000.jpg
```

#### Multiple Controls (1 Target + 3 Controls)

```
1. Click on front.jpg (Target)
2. Click on left.jpg (Control #1)
3. Click on right.jpg (Control #2)
4. Click on back.jpg (Control #3)
5. Press 7 (saves to dataset 7)
```

**Result:**

```
datasets/7/
  ├── target/000000.jpg
  ├── control_1/000000.jpg
  ├── control_2/000000.jpg
  └── control_3/000000.jpg
```

**Important:** Dataset 7 now **requires** 1 target + 3 controls for all future saves!

#### Adding More to Existing Dataset

```
Same structure as established:
1. Select 1 target
2. Select 3 controls (same count as first save)
3. Press 7
4. ✓ Saved as 000001.jpg
```

**Error Example:**

```
Try to save with wrong structure:
1. Select 1 target
2. Select 2 controls (wrong - needs 3!)
3. Press 7
❌ Error: "Dataset 7 requires 1 target + 3 controls.
           You selected 1 target + 2 controls.
           Please add 1 control(s)."
```

### Comparing Images (Optional)

#### Method 1: Before Saving

```
1. Select target + control(s)
2. Press C (opens comparison modal)
3. Switch between modes
4. Press 0-9 to save from modal
5. Modal auto-closes
```

#### Method 2: After Selecting

```
1. Select images
2. Click "Compare (C)" button
3. Review images
4. Press 0-9 or click dataset button
```

### Comparison Modes

#### 📐 Side by Side

- Shows all images horizontally
- No cropping or distortion
- Scroll left/right to see all
- Each image labeled (Target, Control #1, etc.)
- **Best for**: Quick visual comparison

#### 🔀 Overlay Slider

- Drag slider to reveal target vs control
- Click anywhere to move slider
- Perfect alignment (same dimensions)
- Switch between multiple controls with buttons
- **Best for**: Detailed pixel-level comparison

#### 🎨 Blend

- Overlay with opacity control
- Slider: 0% (target) → 50% (default) → 100% (control)
- Smooth transitions
- Switch controls with buttons
- **Best for**: Seeing differences in transparency

**Note:** Modal height is fixed (85vh) - no jumping when switching modes!

### Understanding Visual Indicators

#### Resolution Colors (Top-Right)

- **Gray background**: `2048×1536` - Will be downscaled (good quality)
- **Red background**: `256×256` - Will be upscaled (may lose quality)
- **No color**: Same size as target or unknown

#### Selection Outlines

- **Blue outline**: Target image (always first)
- **Green outline**: Control images (second onwards)
- **Blue label**: "Target" at bottom-left
- **Green label**: "Control #1", "Control #2", etc.

#### Dataset Tags (Bottom-Right)

- **Purple badge**: "D: 5,7" - Image saved in datasets 5 and 7
- Appears after saving
- Shows all datasets containing this image

### Keyboard Shortcuts

| Key   | Action          | Notes                       |
| ----- | --------------- | --------------------------- |
| `0-9` | Save to dataset | Works from grid or modal    |
| `C`   | Open comparison | Requires 2+ selected images |
| `U`   | Unselect all    | Clears all selections       |
| `ESC` | Close modal     | Closes comparison window    |

**Important:** Shortcuts only work **without modifiers**:

- ✅ `C` = Open comparison
- ❌ `Ctrl+C` = Browser copy (not hijacked)
- ❌ `Cmd+C` = Browser copy (not hijacked)

### Dataset Management

#### Viewing Dataset Info

- Dataset buttons show structure below number
- Example: Button "7" with "1+3" means 1 target + 3 controls
- Active datasets highlighted in blue
- Inactive datasets grayed out

#### Dataset Structure Rules

1. **First Save Sets Structure**: Adding 1+2 to dataset 3 locks it to 1+2
2. **All Saves Must Match**: Every future save must have 1+2
3. **Validation on Save**: Server checks structure before saving
4. **Clear Errors**: Tells you exactly what to add/remove

#### File Naming

- Sequential: `000000.jpg`, `000001.jpg`, ..., `999999.jpg`
- Target and controls use same filename
- Up to 1 million pairs per dataset

## 🎨 Image Conversion Process

### How Smart Scaling Works

```
Original Image: 1920×1080 PNG
Target Size: 512×512 JPEG

Step 1: Calculate scale to cover target
  scale = max(512/1920, 512/1080) = 0.474

Step 2: Scale image
  new_width = 1920 × 0.474 = 910px
  new_height = 1080 × 0.474 = 512px

Step 3: Center crop to exact size
  crop_left = (910 - 512) / 2 = 199px
  crop_right = 199px
  Result: 512×512

Step 4: Convert to JPEG
  Quality: 95%
  Chroma: 4:4:4
  Format: JPEG (forced)
```

### Quality Settings

**JPEG Encoding:**

- Quality: 95% (near-lossless)
- Chroma Subsampling: 4:4:4 (highest quality)
- Color Space: RGB
- Format: Baseline JPEG with JFIF headers

**Why These Settings:**

- macOS requires proper JFIF headers to display images
- 95% quality preserves visual quality while reducing file size
- 4:4:4 prevents color artifacts in gradients
- Baseline JPEG ensures universal compatibility

## 📁 Dataset Structure

### Output Directory Structure

```
datasets/
├── 0/
│   ├── target/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── 000002.jpg
│   ├── control_1/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── 000002.jpg
│   └── control_2/
│       ├── 000000.jpg
│       ├── 000001.jpg
│       └── 000002.jpg
├── 1/
│   ├── target/
│   └── control_1/
├── 2/
├── ...
└── 9/
```

### Structure Validation

**Server Endpoint:**

```javascript
GET /api/dataset/:id/structure

Response:
{
  "exists": true,
  "structure": {
    "target": 1,
    "controls": 3
  }
}
```

**Frontend Validation:**

```javascript
// Check before save
if (dataset.structure && dataset.structure.controls !== selectedControls) {
  alert("Structure mismatch!");
  return;
}
```

**Server Validation:**

```javascript
// Double-check on server
if (existingStructure && existingStructure !== newStructure) {
  return 400 Bad Request;
}
```

## 🔧 Technical Details

### Frontend Architecture

**Technologies:**

- Vanilla JavaScript (no framework dependencies)
- Tailwind CSS (CDN)
- Masonry.js (responsive grid layout)
- Canvas API (image processing)

**Key Components:**

- `convertImageToTargetSize()` - Client-side image conversion
- `renderPage()` - Optimized rendering with placeholders
- `saveToDataset()` - Validation and save workflow
- `renderComparison()` - Comparison mode rendering

**Performance Features:**

- Image caching with Map
- Debounced masonry layout
- Document fragments for DOM insertion
- Lazy image loading
- Scroll position tracking

### Backend Architecture

**Technologies:**

- Express 5.2.1 (web server)
- Sharp 0.33.5 (image processing)
- Multer 2.0.2 (file uploads)

**Key Endpoints:**

```javascript
GET  /                              // Serve HTML
GET  /api/dataset/:id/structure     // Get dataset structure
POST /api/dataset/save              // Save dataset with validation
GET  /api/datasets/stats            // Get all dataset counts
POST /api/images/folder             // Upload folder images
POST /api/images/urls               // Process URL list
```

**Sharp Processing:**

```javascript
await sharp(buffer)
  .jpeg({
    quality: 95,
    chromaSubsampling: "4:4:4",
    force: true,
  })
  .toFile(filepath);
```

### Data Flow

```
User selects images
  ↓
Frontend converts to target size (Canvas API)
  ↓
Frontend validates structure (if dataset exists)
  ↓
Frontend sends base64 JPEG to server
  ↓
Server validates structure (double-check)
  ↓
Server uses Sharp for final encoding
  ↓
Server saves to datasets/N/target|control_X/NNNNNN.jpg
  ↓
Server returns success
  ↓
Frontend updates UI (maintains scroll position)
```

## 🎯 Best Practices

### For Best Results

1. **Set Target Size First**

   - Choose resolution before loading images
   - Changing size later requires regenerating previews

2. **Use Consistent Source Images**

   - Similar resolutions for best quality
   - Avoid mixing very small and very large images

3. **Match Dataset Structure**

   - First save defines structure
   - Plan your control count in advance

4. **Compare Before Saving**

   - Use comparison modes to verify pairs
   - Ensure target and controls are correctly matched

5. **Monitor Resolution Indicators**
   - Gray = good (downscaling)
   - Red = warning (upscaling may reduce quality)

### Dataset Organization

**Recommended Structure:**

```
Dataset 0: Simple pairs (1+1)
Dataset 1: Simple pairs (1+1)
Dataset 2: Multi-angle (1+3) - front + left/right/back
Dataset 3: Multi-angle (1+3)
Dataset 4: Variations (1+5) - base + 5 variations
...
```

### Performance Tips

**For Large Image Sets (1,000+):**

- Process in batches of 500-1,000
- Use smaller preview sizes during selection
- Close comparison modal when not needed
- Clear browser cache if slow

**For High-Resolution Outputs:**

- Use 1024×1024 or higher for training
- Ensure source images are at least that size
- Monitor disk space (JPEG files are smaller but still substantial)

## 📊 Dataset Quality Assurance

### What Makes a High-Quality Dataset

✅ **Consistent Dimensions**: All images exactly same size
✅ **Proper Aspect Ratios**: No distortion from stretching
✅ **Center-Focused**: Important content not cropped
✅ **High JPEG Quality**: 95% preserves visual detail
✅ **Matched Pairs**: Target and controls properly aligned
✅ **Sequential Naming**: Easy to track and manage

### Quality Checks

**Before Saving:**

1. Use comparison modes to verify pairs
2. Check resolution indicators (avoid red if possible)
3. Ensure structure matches dataset requirements
4. Preview images in grid (shows actual output)

**After Saving:**

1. Check dataset folder for expected structure
2. Open a few JPEGs to verify quality
3. Verify file sizes are reasonable (not 0 bytes)
4. Count files matches expected count

## 🔄 Workflow Examples

### Example 1: Creating Facial Expressions Dataset

**Goal:** Train model on different facial expressions

```
Dataset 3: Neutral → Smiling

1. Load folder with portrait photos
2. Settings → 512×512
3. Select neutral_face_01.jpg (target)
4. Select smiling_face_01.jpg (control)
5. Press 3 → Saved!
6. Repeat for all pairs
7. Result: 100 pairs in dataset 3
```

### Example 2: Multi-Angle Object Dataset

**Goal:** Different views of same object

```
Dataset 5: Object from multiple angles

1. Load object photos
2. Settings → 1024×1024
3. Select front_view.jpg (target)
4. Select left_view.jpg (control #1)
5. Select right_view.jpg (control #2)
6. Select back_view.jpg (control #3)
7. Press 5 → Saved with 1+3 structure!
8. All future saves to dataset 5 must have 1+3
```

### Example 3: Style Transfer Dataset

**Goal:** Original → Stylized versions

```
Dataset 8: Photo → Artistic styles

1. Select photo_01.jpg (target)
2. Select oil_painting_01.jpg (control #1)
3. Select watercolor_01.jpg (control #2)
4. Select sketch_01.jpg (control #3)
5. Select cartoon_01.jpg (control #4)
6. Press 8 → Saved with 1+4 structure!
```

---

**Made with ❤️ for high-quality ML dataset creation**
