/**
 * CONFIGURATION
 * Centralized settings for application behavior.
 */
const CONFIG = {
  WARMUP_IMAGE_BASE64:
    "data:image/jpeg;base64,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", // PASTE A VALID JPG BASE64 STRING HERE TO ENABLE AUTO WARMUP
  ITEMS_PER_PAGE: 20,
  ALIGNMENT_ITERATIONS: 5,
  MAX_IMAGE_SIZE: 1280,
  DELAY_MS: 10,
};

/**
 * APPLICATION STATE
 * Manages runtime data to avoid global scope pollution.
 */
const STATE = {
  files: {
    target: new Map(),
    control: new Map(),
  },
  pairs: [],
  selection: new Set(),
  page: 1,
  isBatchCancelled: false,
  editor: {
    id: null,
    pos: { x: 0, y: 0, startX: 0, startY: 0 },
    images: { target: null, control: null },
  },
  ai: {
    instance: null,
    queue: [], // FIFO queue for MediaPipe requests
  },
};

/* --- UTILITIES --- */

function log(msg, type = "info") {
  const panel = document.getElementById("debug-panel");
  const entry = document.createElement("div");
  entry.className = `debug-entry debug-${type}`;

  let icon = "ℹ";
  if (type === "success") icon = "✓";
  if (type === "error") icon = "✗";
  if (type === "warn") icon = "⚠";

  entry.innerHTML = `
                <span style="flex:1; display:flex; align-items:center; gap:8px;">
                    <span style="font-weight:bold; width:20px;">${icon}</span> ${msg}
                </span>
                <span class="debug-cmd">${new Date().toLocaleTimeString()}</span>
            `;

  panel.appendChild(entry);
  panel.scrollTop = panel.scrollHeight;
  console.log(`[${type.toUpperCase()}] ${msg}`);
}

function toggleDebug() {
  const panel = document.getElementById("debug-panel");
  const btn = document.getElementById("debug-toggle");
  const isOpen = panel.style.display === "block";
  panel.style.display = isOpen ? "none" : "block";
  btn.classList.toggle("visible", !isOpen);
}

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.shiftKey && e.key === "L") {
    e.preventDefault();
    toggleDebug();
  }
});

function showToast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = "toast";
  el.style.borderLeftColor =
    type === "error"
      ? "var(--error)"
      : type === "success"
      ? "var(--success)"
      : "#fff";
  el.textContent = msg;
  document.getElementById("toast-container").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 300);
  }, 3000);
}

/* --- AI & ALIGNMENT LOGIC --- */

async function initMediaPipe() {
  if (STATE.ai.instance) return STATE.ai.instance;

  log("Initializing Face Mesh Model...", "info");
  const FaceMesh = window.FaceMesh;

  STATE.ai.instance = new FaceMesh({
    locateFile: (file) => {
      if (window.MEDIAPIPE_URLS && window.MEDIAPIPE_URLS[file]) {
        return window.MEDIAPIPE_URLS[file];
      }
      return `js/vendor/${file}`;
    },
  });

  STATE.ai.instance.setOptions({
    maxNumFaces: 1,
    refineLandmarks: false,
    minDetectionConfidence: 0.2,
    minTrackingConfidence: 0.2,
  });

  // FIFO Queue Handler: Resolves the oldest promise in the queue
  STATE.ai.instance.onResults((results) => {
    const resolve = STATE.ai.queue.shift();
    if (resolve) resolve(results);
  });

  try {
    await STATE.ai.instance.initialize();
    log("Model loaded successfully.", "success");
  } catch (e) {
    log(`Initialization Failed: ${e}`, "error");
    throw e;
  }

  return STATE.ai.instance;
}

// Warm-up ensures the WASM model is fully active before user interaction
async function performWarmUp() {
  if (!CONFIG.WARMUP_IMAGE_BASE64) {
    log("Warm-up skipped: No image configured.", "warn");
    return;
  }

  log("Starting COMPLETE Engine Warm-up (Full Analog Simulation)...", "info");

  const warmupImg = new Image();
  warmupImg.src = CONFIG.WARMUP_IMAGE_BASE64;

  try {
    // 1. Load the image into memory
    await new Promise((resolve, reject) => {
      warmupImg.onload = resolve;
      warmupImg.onerror = () =>
        reject(new Error("Failed to load warmup image"));
    });

    // 2. Initialize MediaPipe (loads WASM)
    await initMediaPipe();

    // 3. Execute FULL Alignment Pipeline
    // We pass the same image as both Target and Control to simulate a "real" alignment job.
    // This runs the detection loop, the cropping logic, and blob generation.
    const result = await runFullAlignment(warmupImg, warmupImg);

    // 4. Verify Success: Ensure the process actually generated output data
    if (result && result.tBlob && result.cBlob) {
      log(
        "Warm-up Complete: Full processing pipeline verified successfully.",
        "success"
      );
    } else {
      log("Warm-up Failed: Pipeline executed but returned no data.", "error");
      throw new Error("Pipeline verification failed");
    }
  } catch (e) {
    log(`Warm-up Critical Error: ${e}`, "error");
    showToast("AI Warm-up failed. Check logs.", "error");
  }
}

async function getFaceCenter(imgElement) {
  if (!STATE.ai.instance) await initMediaPipe();

  const origW = imgElement.naturalWidth;
  const origH = imgElement.naturalHeight;
  if (origW === 0 || origH === 0) return null;

  // Scale down large images for performance
  let scale = 1;
  if (origW > CONFIG.MAX_IMAGE_SIZE || origH > CONFIG.MAX_IMAGE_SIZE) {
    scale = CONFIG.MAX_IMAGE_SIZE / Math.max(origW, origH);
  }

  const canvas = document.createElement("canvas");
  canvas.width = origW * scale;
  canvas.height = origH * scale;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

  // Queue Request
  const p = new Promise((resolve) => STATE.ai.queue.push(resolve));
  STATE.ai.instance.send({ image: canvas });
  const results = await p;

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    const landmarks = results.multiFaceLandmarks[0];
    let sumX = 0,
      sumY = 0;
    for (const lm of landmarks) {
      sumX += lm.x;
      sumY += lm.y;
    }
    return {
      x: ((sumX / landmarks.length) * origW) / scale,
      y: ((sumY / landmarks.length) * origH) / scale,
    };
  }
  return null;
}

function loadImageFromBlob(blob) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = URL.createObjectURL(blob);
    img.onload = () => resolve(img);
    img.onerror = reject;
  });
}

// Crop and align two images based on calculated delta (dx, dy)
async function performCrop(tImg, cImg, dx, dy) {
  const origW = tImg.naturalWidth;
  const origH = tImg.naturalHeight;

  const cropLeft = Math.max(0, dx);
  const cropRight = Math.max(0, -dx);
  const cropTop = Math.max(0, dy);
  const cropBottom = Math.max(0, -dy);

  const newW = origW - cropLeft - cropRight;
  const newH = origH - cropTop - cropBottom;

  if (newW <= 0 || newH <= 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = newW;
  canvas.height = newH;
  const ctx = canvas.getContext("2d");

  // Draw Target
  ctx.drawImage(tImg, cropLeft, cropTop, newW, newH, 0, 0, newW, newH);
  const tBlob = await new Promise((r) => canvas.toBlob(r, "image/jpeg"));

  // Draw Control (with offset)
  ctx.clearRect(0, 0, newW, newH);
  ctx.drawImage(
    cImg,
    cropLeft - dx,
    cropTop - dy,
    newW,
    newH,
    0,
    0,
    newW,
    newH
  );
  const cBlob = await new Promise((r) => canvas.toBlob(r, "image/jpeg"));

  return { tBlob, cBlob, newW, newH };
}

// Iterative refinement loop for high precision alignment
async function runFullAlignment(tImg, cImg) {
  let workingT = tImg;
  let workingC = cImg;

  for (let i = 0; i < CONFIG.ALIGNMENT_ITERATIONS; i++) {
    if (STATE.isBatchCancelled) return null;
    await new Promise((r) => setTimeout(r, CONFIG.DELAY_MS));

    const tCenter = await getFaceCenter(workingT);
    const cCenter = await getFaceCenter(workingC);

    if (!tCenter || !cCenter) {
      log("Face lost during iteration.", "error");
      return null;
    }

    const dx = tCenter.x - cCenter.x;
    const dy = tCenter.y - cCenter.y;

    const result = await performCrop(workingT, workingC, dx, dy);
    if (!result) {
      log("Crop failed (no overlap).", "error");
      return null;
    }

    // Reload blobs into Images for next pass
    workingT = await loadImageFromBlob(result.tBlob);
    workingC = await loadImageFromBlob(result.cBlob);
  }
  // Final clean crop
  return await performCrop(workingT, workingC, 0, 0);
}

/* --- FILE HANDLING --- */

function handleFolderSelect(type, input) {
  const files = Array.from(input.files).filter((f) =>
    f.type.startsWith("image/")
  );
  STATE.files[type].clear();

  if (files.length === 0) return showToast("No images found", "error");
  files.forEach((f) => STATE.files[type].set(f.name.toLowerCase(), f));

  document.getElementById(`card-${type}`).classList.add("selected");
  document.getElementById(
    `status-${type}`
  ).textContent = `${files.length} loaded`;

  const startBtn = document.getElementById("start-btn");
  startBtn.disabled = !(
    STATE.files.target.size > 0 && STATE.files.control.size > 0
  );
}

function startComparison() {
  STATE.pairs = [];
  for (const [name, tFile] of STATE.files.target) {
    if (STATE.files.control.has(name)) {
      STATE.pairs.push({
        id: name.split(".")[0],
        targetFile: tFile,
        controlFile: STATE.files.control.get(name),
      });
    }
  }

  if (STATE.pairs.length === 0)
    return showToast("No matching filenames", "error");

  document.getElementById("landing-view").classList.add("hidden");
  document.getElementById("viewer-view").style.display = "flex";
  renderGrid();
}

/* --- UI RENDERING --- */

function renderGrid() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  const start = (STATE.page - 1) * CONFIG.ITEMS_PER_PAGE;
  const end = start + CONFIG.ITEMS_PER_PAGE;
  const pagePairs = STATE.pairs.slice(start, end);

  if (pagePairs.length === 0) {
    grid.innerHTML =
      '<div style="grid-column:1/-1;text-align:center;padding:40px;color:#666">No images.</div>';
    return;
  }

  pagePairs.forEach((pair) => {
    const card = document.createElement("div");
    card.className = `card ${STATE.selection.has(pair.id) ? "selected" : ""}`;
    card.dataset.id = pair.id;

    // Hover logic
    card.onmouseenter = (e) => {
      if (e.target.closest(".card-actions")) return;
      card.classList.add("hover-active");
    };
    card.onmouseleave = (e) => {
      if (e.relatedTarget && e.relatedTarget.closest(".card-actions")) {
        card.classList.add("hover-actions-active");
      } else {
        card.classList.remove("hover-active", "hover-actions-active");
      }
    };

    card.onclick = (e) => {
      if (e.target.closest(".card-actions")) return;
      toggleSelection(pair.id, card);
    };

    card.innerHTML = `
                    <div class="select-check"><svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="3" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"></path></svg></div>
                    <div class="card-actions" onmouseenter="this.closest('.card').classList.add('hover-actions-active')" onmouseleave="this.closest('.card').classList.remove('hover-actions-active')">
                        <div class="action-btn btn-quick" onclick="quickAutoAlign('${
                          pair.id
                        }', this)" title="Auto Align"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg></div>
                        <div class="action-btn btn-edit" onclick="openEditor('${
                          pair.id
                        }')" title="Edit"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-3a2.121 2.121 0 01-3-3l9.5-9.5z"></path></svg></div>
                        <div class="action-btn btn-delete" onclick="deletePair('${
                          pair.id
                        }')" title="Delete"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg></div>
                    </div>
                    <div class="card-id">${pair.id}</div>
                    <div class="img-container">
                        <img id="img-target-${
                          pair.id
                        }" src="${URL.createObjectURL(
      pair.targetFile
    )}" class="img-layer img-target" loading="lazy">
                        <img id="img-control-${
                          pair.id
                        }" src="${URL.createObjectURL(
      pair.controlFile
    )}" class="img-layer img-control" loading="lazy">
                    </div>
                `;
    grid.appendChild(card);
  });

  // Pagination Updates
  const totalPages = Math.ceil(STATE.pairs.length / CONFIG.ITEMS_PER_PAGE) || 1;
  document.getElementById(
    "pageInfo"
  ).textContent = `${STATE.page} / ${totalPages}`;
  document.getElementById("prevBtn").disabled = STATE.page === 1;
  document.getElementById("nextBtn").disabled = STATE.page === totalPages;
}

function changePage(delta) {
  const totalPages = Math.ceil(STATE.pairs.length / CONFIG.ITEMS_PER_PAGE);
  const newPage = STATE.page + delta;
  if (newPage >= 1 && newPage <= totalPages) {
    STATE.page = newPage;
    renderGrid();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

/* --- DATA MANIPULATION --- */

function toggleSelection(id, card) {
  if (STATE.selection.has(id)) {
    STATE.selection.delete(id);
    card.classList.remove("selected");
  } else {
    STATE.selection.add(id);
    card.classList.add("selected");
  }
  updateHeaderStats();
}

function clearSelection() {
  STATE.selection.clear();
  renderGrid();
  updateHeaderStats();
}

function deletePair(id) {
  if (!confirm("Delete this pair?")) return;
  STATE.pairs = STATE.pairs.filter((p) => p.id !== id);
  STATE.selection.delete(id);
  renderGrid();
  updateHeaderStats();
  showToast("Pair deleted");
}

function updateHeaderStats() {
  const count = STATE.selection.size;
  document.getElementById("selected-count").textContent = count;
  document.getElementById("btn-download").disabled = count === 0;
}

async function quickAutoAlign(id, btn) {
  if (btn.disabled) return;
  const pair = STATE.pairs.find((p) => p.id === id);
  if (!pair) return;

  btn.disabled = true;
  btn.innerHTML = `<svg width="16" height="16" class="animate-spin" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"></path></svg>`;

  try {
    const [tImg, cImg] = await Promise.all([
      loadImageFromBlob(pair.targetFile),
      loadImageFromBlob(pair.controlFile),
    ]);

    const result = await runFullAlignment(tImg, cImg);

    if (result) {
      pair.targetFile = new File([result.tBlob], pair.targetFile.name, {
        type: "image/jpeg",
      });
      pair.controlFile = new File([result.cBlob], pair.controlFile.name, {
        type: "image/jpeg",
      });

      if (!STATE.selection.has(id)) {
        STATE.selection.add(id);
        const card = document.querySelector(`.card[data-id="${id}"]`);
        if (card) card.classList.add("selected");
      }

      document.getElementById(`img-target-${id}`).src = URL.createObjectURL(
        pair.targetFile
      );
      document.getElementById(`img-control-${id}`).src = URL.createObjectURL(
        pair.controlFile
      );

      showToast(`Aligned ${id}`, "success");
    } else {
      showToast("Alignment failed", "error");
    }
  } catch (e) {
    log(`Error: ${e}`, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>`;
  }
}

/* --- EDITOR MODAL --- */

function openEditor(id) {
  STATE.editor.id = id;
  const pair = STATE.pairs.find((p) => p.id === id);
  if (!pair) return;

  const workspace = document.getElementById("workspace");
  workspace.innerHTML = "";

  const imgBase = document.createElement("img");
  imgBase.src = URL.createObjectURL(pair.targetFile);
  imgBase.className = "editor-img editor-base";

  const imgOverlay = document.createElement("img");
  imgOverlay.src = URL.createObjectURL(pair.controlFile);
  imgOverlay.className = "editor-img editor-overlay";

  workspace.appendChild(imgBase);
  workspace.appendChild(imgOverlay);

  STATE.editor.images = { target: imgBase, control: imgOverlay };
  resetEditor();

  imgBase.onload = () => {
    centerImage(imgBase);
    centerImage(imgOverlay);
  };

  document.getElementById("edit-modal").style.display = "flex";
}

function centerImage(img) {
  const ws = document.getElementById("workspace").getBoundingClientRect();
  const scale =
    Math.min(ws.height / img.naturalHeight, ws.width / img.naturalWidth) * 0.9;
  img.style.width = `${img.naturalWidth * scale}px`;
  img.style.height = `${img.naturalHeight * scale}px`;

  const x = (ws.width - img.offsetWidth) / 2;
  const y = (ws.height - img.offsetHeight) / 2;

  if (img.classList.contains("editor-base")) {
    img.style.left = `${x}px`;
    img.style.top = `${y}px`;
  } else {
    img.style.left = STATE.editor.images.target.style.left;
    img.style.top = STATE.editor.images.target.style.top;
  }
}

function resetEditor() {
  STATE.editor.pos = { x: 0, y: 0, startX: 0, startY: 0 };
  updateOverlayPos();
}

function updateOverlayPos() {
  const base = STATE.editor.images.target;
  const overlay = STATE.editor.images.control;
  if (!base || !overlay) return;

  const bx = parseFloat(base.style.left);
  const by = parseFloat(base.style.top);

  overlay.style.left = `${bx + STATE.editor.pos.x}px`;
  overlay.style.top = `${by + STATE.editor.pos.y}px`;

  document.getElementById("off-x").textContent = Math.round(STATE.editor.pos.x);
  document.getElementById("off-y").textContent = Math.round(STATE.editor.pos.y);
}

async function autoAlignSingle() {
  const btn = document.getElementById("btn-auto-single");
  const original = btn.innerHTML;
  btn.innerHTML = `<svg class="animate-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"></path></svg>`;
  btn.disabled = true;

  try {
    const result = await runFullAlignment(
      STATE.editor.images.target,
      STATE.editor.images.control
    );
    if (result) {
      const [newT, newC] = await Promise.all([
        loadImageFromBlob(result.tBlob),
        loadImageFromBlob(result.cBlob),
      ]);

      const ws = document.getElementById("workspace");
      ws.innerHTML = "";
      ws.appendChild(newT);
      ws.appendChild(newC);

      newT.className = "editor-img editor-base";
      newC.className = "editor-img editor-overlay";
      STATE.editor.images = { target: newT, control: newC };
      resetEditor();

      centerImage(newT);
      centerImage(newC);
      showToast("Aligned successfully", "success");
    }
  } catch (e) {
    log(`Editor Error: ${e}`, "error");
  } finally {
    btn.innerHTML = original;
    btn.disabled = false;
  }
}

// Drag Logic
const workspace = document.getElementById("workspace");
workspace.addEventListener("mousedown", (e) => {
  if (e.target !== workspace && !e.target.classList.contains("editor-img"))
    return;
  STATE.editor.pos.startX = e.clientX - STATE.editor.pos.x;
  STATE.editor.pos.startY = e.clientY - STATE.editor.pos.y;
  workspace.addEventListener("mousemove", onDrag);
  workspace.addEventListener("mouseup", stopDrag);
});

function onDrag(e) {
  e.preventDefault();
  STATE.editor.pos.x = e.clientX - STATE.editor.pos.startX;
  STATE.editor.pos.y = e.clientY - STATE.editor.pos.startY;
  updateOverlayPos();
}

function stopDrag() {
  workspace.removeEventListener("mousemove", onDrag);
  workspace.removeEventListener("mouseup", stopDrag);
}

function closeModal() {
  document.getElementById("edit-modal").style.display = "none";
  document.getElementById("workspace").innerHTML = "";
  STATE.editor.id = null;
}

async function saveAlignment() {
  const id = STATE.editor.id;
  if (!id) return;
  const pairIdx = STATE.pairs.findIndex((p) => p.id === id);
  if (pairIdx === -1) return;

  const tImg = STATE.editor.images.target;
  const cImg = STATE.editor.images.control;

  const baseRect = tImg.getBoundingClientRect();
  const scaleFactor = baseRect.width / tImg.naturalWidth;
  const realDx = STATE.editor.pos.x / scaleFactor;
  const realDy = STATE.editor.pos.y / scaleFactor;

  const result = await performCrop(tImg, cImg, realDx, realDy);
  if (result) {
    STATE.pairs[pairIdx].targetFile = new File(
      [result.tBlob],
      STATE.pairs[pairIdx].targetFile.name,
      { type: "image/jpeg" }
    );
    STATE.pairs[pairIdx].controlFile = new File(
      [result.cBlob],
      STATE.pairs[pairIdx].controlFile.name,
      { type: "image/jpeg" }
    );
    showToast(`Saved! ${result.newW}x${result.newH}`, "success");
    closeModal();
    renderGrid();
  } else {
    showToast("Offset invalid", "error");
  }
}

/* --- BATCH PROCESSING --- */

async function startBatchAlignment() {
  await initMediaPipe();

  const modal = document.getElementById("batch-modal");
  const list = document.getElementById("batch-list");
  const bar = document.getElementById("batch-progress");

  modal.style.display = "flex";
  list.innerHTML = "";
  STATE.isBatchCancelled = false;

  STATE.pairs.forEach((pair) => {
    const item = document.createElement("div");
    item.className = "batch-item";
    item.id = `batch-${pair.id}`;
    item.innerHTML = `<span>${pair.id}</span><span class="batch-status">⏳</span>`;
    list.appendChild(item);
  });

  const total = STATE.pairs.length;
  let processed = 0;

  for (const pair of STATE.pairs) {
    if (STATE.isBatchCancelled) break;
    const item = document.getElementById(`batch-${pair.id}`);
    item.classList.add("processing");
    item.querySelector(".batch-status").textContent = "...";
    await new Promise((r) => setTimeout(r, CONFIG.DELAY_MS));

    try {
      const [tImg, cImg] = await Promise.all([
        loadImageFromBlob(pair.targetFile),
        loadImageFromBlob(pair.controlFile),
      ]);

      const result = await runFullAlignment(tImg, cImg);

      if (result) {
        pair.targetFile = new File([result.tBlob], pair.targetFile.name, {
          type: "image/jpeg",
        });
        pair.controlFile = new File([result.cBlob], pair.controlFile.name, {
          type: "image/jpeg",
        });
        item.classList.replace("processing", "success");
        item.querySelector(".batch-status").innerHTML = "✓";
      } else {
        throw new Error("Fail");
      }
    } catch (e) {
      item.classList.replace("processing", "error");
      item.querySelector(".batch-status").innerHTML = "✗";
    }

    processed++;
    bar.style.width = `${(processed / total) * 100}%`;
    document.getElementById(
      "batch-status-text"
    ).textContent = `${processed} / ${total}`;
  }

  // Select all successful items
  STATE.selection.clear();
  STATE.pairs.forEach((pair) => {
    if (
      document.getElementById(`batch-${pair.id}`)?.classList.contains("success")
    ) {
      STATE.selection.add(pair.id);
    }
  });

  modal.style.display = "none";
  renderGrid();
  showToast(`Batch Complete. ${STATE.selection.size} aligned.`, "success");
}

function cancelBatch() {
  STATE.isBatchCancelled = true;
}

/* --- EXPORT --- */

function exportDataset() {
  if (STATE.selection.size === 0) return;
  const zip = new JSZip();
  const fT = zip.folder("target");
  const fC = zip.folder("control");

  const filesToZip = [];
  STATE.pairs.forEach((p) => {
    if (STATE.selection.has(p.id)) filesToZip.push(p.targetFile, p.controlFile);
  });

  const btn = document.getElementById("btn-download");
  const originalText = btn.innerHTML;
  btn.innerHTML = "Compressing...";
  btn.disabled = true;

  const promises = filesToZip.map(
    (file) =>
      new Promise((resolve) => {
        const reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = (e) => {
          const pair = STATE.pairs.find(
            (p) => p.targetFile === file || p.controlFile === file
          );
          const folder = pair && pair.targetFile === file ? fT : fC;
          folder.file(file.name, e.target.result);
          resolve();
        };
      })
  );

  Promise.all(promises).then(() => {
    zip.generateAsync({ type: "blob" }).then((content) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(content);
      a.download = `lora_dataset_${Date.now()}.zip`;
      a.click();
      showToast("Download started!");
      btn.innerHTML = originalText;
      btn.disabled = false;
    });
  });
}

/* --- INITIALIZATION --- */

document.addEventListener("DOMContentLoaded", async () => {
  // Start warmup automatically on load
  await performWarmUp();
});
