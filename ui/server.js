const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs").promises;
const fsSync = require("fs");
const cors = require("cors");
const sharp = require("sharp");

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: "100mb" }));
app.use(express.static("public"));

const datasetsDir = path.join(__dirname, "datasets");
if (!fsSync.existsSync(datasetsDir)) {
  fsSync.mkdirSync(datasetsDir, { recursive: true });
}

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, "uploads");
    if (!fsSync.existsSync(uploadDir)) {
      fsSync.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage: storage });

app.post("/api/images/folder", upload.array("images"), async (req, res) => {
  try {
    const images = req.files.map((file) => ({
      url: `/uploads/${file.filename}`,
      name: file.originalname,
      size: `${file.size}`,
      path: file.path,
    }));
    res.json({ images });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/images/urls", async (req, res) => {
  try {
    const { urls } = req.body;
    const images = urls.map((url, idx) => ({
      url: url.trim(),
      name: url.split("/").pop() || `image_${idx}`,
      size: "N/A",
    }));
    res.json({ images });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get dataset structure (number of controls)
app.get("/api/dataset/:id/structure", async (req, res) => {
  try {
    const datasetId = req.params.id;
    const datasetPath = path.join(datasetsDir, datasetId);

    if (!fsSync.existsSync(datasetPath)) {
      return res.json({ exists: false });
    }

    const subdirs = await fs.readdir(datasetPath);
    const controlDirs = subdirs.filter((d) => d.startsWith("control_")).sort();

    res.json({
      exists: true,
      structure: {
        target: 1,
        controls: controlDirs.length,
      },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Save dataset with structure validation
app.post("/api/dataset/save", async (req, res) => {
  try {
    const { datasetId, images } = req.body;

    const datasetPath = path.join(datasetsDir, datasetId.toString());
    const targetPath = path.join(datasetPath, "target");

    // Check existing structure
    const controlCount = images.length - 1;
    if (fsSync.existsSync(datasetPath)) {
      const subdirs = await fs.readdir(datasetPath);
      const existingControlDirs = subdirs
        .filter((d) => d.startsWith("control_"))
        .sort();

      if (
        existingControlDirs.length > 0 &&
        existingControlDirs.length !== controlCount
      ) {
        return res.status(400).json({
          error: "Structure mismatch",
          message: `Dataset ${datasetId} requires 1 target + ${existingControlDirs.length} controls`,
          expected: existingControlDirs.length,
          received: controlCount,
        });
      }
    }

    await fs.mkdir(targetPath, { recursive: true });

    const controlPaths = [];
    for (let i = 0; i < controlCount; i++) {
      const controlPath = path.join(datasetPath, `control_${i + 1}`);
      await fs.mkdir(controlPath, { recursive: true });
      controlPaths.push(controlPath);
    }

    const existingFiles = await fs.readdir(targetPath).catch(() => []);
    const nextIndex = existingFiles.filter((f) => f.endsWith(".jpg")).length;
    const filename = String(nextIndex).padStart(6, "0") + ".jpg";

    await saveBase64Image(images[0].data, path.join(targetPath, filename));

    for (let i = 1; i < images.length; i++) {
      await saveBase64Image(
        images[i].data,
        path.join(controlPaths[i - 1], filename)
      );
    }

    res.json({
      success: true,
      filename,
      index: nextIndex,
    });
  } catch (error) {
    console.error("Save error:", error);
    res.status(500).json({ error: error.message });
  }
});

async function saveBase64Image(base64Data, filepath) {
  try {
    const base64String = base64Data.replace(/^data:image\/\w+;base64,/, "");
    const buffer = Buffer.from(base64String, "base64");

    await sharp(buffer)
      .jpeg({
        quality: 95,
        chromaSubsampling: "4:4:4",
        force: true,
      })
      .toFile(filepath);
  } catch (error) {
    console.warn("Sharp processing failed, using direct write:", error.message);
    const base64String = base64Data.replace(/^data:image\/\w+;base64,/, "");
    const buffer = Buffer.from(base64String, "base64");
    await fs.writeFile(filepath, buffer);
  }
}

const uploadsDir = path.join(__dirname, "uploads");
if (!fsSync.existsSync(uploadsDir)) {
  fsSync.mkdirSync(uploadsDir, { recursive: true });
}
app.use("/uploads", express.static(uploadsDir));

app.get("/api/datasets/stats", async (req, res) => {
  try {
    const stats = {};
    for (let i = 0; i <= 9; i++) {
      const datasetPath = path.join(datasetsDir, i.toString(), "target");
      try {
        const files = await fs.readdir(datasetPath);
        stats[i] = files.filter((f) => f.endsWith(".jpg")).length;
      } catch {
        stats[i] = 0;
      }
    }
    res.json(stats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Datasets will be saved to: ${datasetsDir}`);
});
