let allImages = [];
let selectedImages = [];
let datasets = Array(10)
  .fill()
  .map(() => ({ active: false, count: 0, structure: null }));
let currentPage = 1;
const imagesPerPage = 50;
let comparisonMode = "overlay";
let masonry = null;
let currentControlIndex = 0;
let targetWidth = 1024;
let targetHeight = 1024;
let imageCache = new Map();
let scrollPosition = 0;
let datasetStructures = {};

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function toggleSettings() {
  const panel = document.getElementById("settingsPanel");
  const overlay = document.getElementById("settingsOverlay");
  const isHidden = panel.classList.contains("hidden");

  if (isHidden) {
    panel.classList.remove("hidden");
    overlay.classList.remove("hidden");
  } else {
    closeSettings();
  }
}

function closeSettings() {
  document.getElementById("settingsPanel").classList.add("hidden");
  document.getElementById("settingsOverlay").classList.add("hidden");
}

async function setPreset(width, height) {
  document
    .querySelectorAll(".preset-btn")
    .forEach((btn) => btn.classList.remove("active"));
  const btn = document.querySelector(`[data-preset="${width}x${height}"]`);
  if (btn) btn.classList.add("active");

  targetWidth = width;
  targetHeight = height;
  document.getElementById("sizeDisplay").textContent = `${width}×${height}`;
  imageCache.clear();

  if (allImages.length > 0) {
    document.getElementById("sizeLoadingOverlay").classList.remove("hidden");
    scrollPosition = window.scrollY;

    await new Promise((resolve) => setTimeout(resolve, 100));
    await renderPage();

    setTimeout(() => {
      document.getElementById("sizeLoadingOverlay").classList.add("hidden");
      window.scrollTo(0, scrollPosition);
    }, 300);
  }
}

async function loadDatasetStructures() {
  for (let i = 0; i <= 9; i++) {
    try {
      const response = await fetch(
        `http://localhost:3000/api/dataset/${i}/structure`
      );
      const data = await response.json();
      if (data.exists) {
        datasetStructures[i] = data.structure;
        datasets[i].structure = data.structure;
        datasets[i].active = true;
        datasets[i].count = 1;
      }
    } catch (error) {
      console.error(`Failed to load structure for dataset ${i}:`, error);
    }
  }
}

async function convertImageToTargetSize(imageUrl, width, height) {
  const cacheKey = `${imageUrl}_${width}_${height}`;
  if (imageCache.has(cacheKey)) {
    return imageCache.get(cacheKey);
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";

    const timeout = setTimeout(() => {
      reject(new Error("Image load timeout"));
    }, 15000); // 15 second timeout

    img.onload = () => {
      clearTimeout(timeout);

      try {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d", { alpha: false });

        const scale = Math.max(width / img.width, height / img.height);
        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;

        const x = (width - scaledWidth) / 2;
        const y = (height - scaledHeight) / 2;

        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
        imageCache.set(cacheKey, dataUrl);
        resolve(dataUrl);
      } catch (error) {
        clearTimeout(timeout);
        reject(error);
      }
    };

    img.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("Failed to load image"));
    };

    img.src = imageUrl;
  });
}

async function getPreviewImage(imageUrl) {
  try {
    const previewSize = 300;
    const aspectRatio = targetWidth / targetHeight;
    const previewWidth =
      aspectRatio >= 1 ? previewSize : previewSize * aspectRatio;
    const previewHeight =
      aspectRatio >= 1 ? previewSize / aspectRatio : previewSize;

    const cacheKey = `preview_${imageUrl}_${Math.round(
      previewWidth
    )}_${Math.round(previewHeight)}`;
    if (imageCache.has(cacheKey)) {
      return imageCache.get(cacheKey);
    }

    const converted = await convertImageToTargetSize(
      imageUrl,
      Math.round(previewWidth),
      Math.round(previewHeight)
    );

    imageCache.set(cacheKey, converted);
    return converted;
  } catch (error) {
    console.error("Preview generation failed:", error);
    return imageUrl; // Fallback to original
  }
}

function initDatasetButtons() {
  const container = document.getElementById("datasetButtons");
  const modalContainer = document.getElementById("modalDatasetButtons");
  container.innerHTML = "";
  modalContainer.innerHTML = "";

  for (let i = 0; i <= 9; i++) {
    container.appendChild(createDatasetButton(i));
    modalContainer.appendChild(createDatasetButton(i));
  }
}

function createDatasetButton(i) {
  const btn = document.createElement("button");
  btn.className = `dataset-btn ${datasets[i].active ? "active" : ""}`;
  btn.disabled = !datasets[i].active;

  let structureText = "";
  if (datasets[i].structure) {
    structureText = `<div class="text-xs text-gray-300 mt-0.5">1+${datasets[i].structure.controls}</div>`;
  } else if (datasets[i].count > 0) {
    structureText = `<div class="text-xs text-gray-300 mt-0.5">${datasets[i].count}</div>`;
  }

  btn.innerHTML = `<div class="text-base">${i}</div>${structureText}`;
  btn.onclick = () => saveToDataset(i);
  return btn;
}

function truncateFilename(name, maxLength = 20) {
  if (name.length <= maxLength) return name;
  const ext = name.split(".").pop();
  const nameWithoutExt = name.substring(0, name.lastIndexOf("."));
  const truncated =
    nameWithoutExt.substring(0, maxLength - ext.length - 4) + "..." + ext;
  return truncated;
}

async function getImageDimensions(url) {
  return new Promise((resolve) => {
    const img = new Image();
    const timeout = setTimeout(() => resolve({ width: 0, height: 0 }), 3000);
    img.onload = () => {
      clearTimeout(timeout);
      resolve({ width: img.width, height: img.height });
    };
    img.onerror = () => {
      clearTimeout(timeout);
      resolve({ width: 0, height: 0 });
    };
    img.src = url;
  });
}

function showLoading(show = true) {
  document.getElementById("loadingScreen").classList.toggle("hidden", !show);
}

function updateLoadingProgress(current, total) {
  document.getElementById(
    "loadingProgress"
  ).textContent = `Processing ${current} of ${total} images...`;
}

async function handleUrlFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  showLoading(true);

  const text = await file.text();
  const urls = text.split("\n").filter((url) => url.trim());

  const batchSize = 50;
  allImages = [];

  for (let i = 0; i < urls.length; i += batchSize) {
    const batch = urls.slice(i, i + batchSize);
    updateLoadingProgress(Math.min(i + batchSize, urls.length), urls.length);

    const batchResults = await Promise.all(
      batch.map(async (url, idx) => {
        const trimmedUrl = url.trim();
        const dims = await getImageDimensions(trimmedUrl);
        return {
          url: trimmedUrl,
          name: trimmedUrl.split("/").pop() || `image_${i + idx}`,
          size: "URL",
          width: dims.width,
          height: dims.height,
          datasets: [],
        };
      })
    );

    allImages.push(...batchResults);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  showLoading(false);
  await showMainInterface();
}

async function handleFolderSelect(event) {
  const files = Array.from(event.target.files);
  const imageFiles = files.filter((f) => f.type.startsWith("image/"));

  if (imageFiles.length === 0) {
    alert("No image files found in the selected folder.");
    return;
  }

  showLoading(true);

  const batchSize = 30;
  allImages = [];

  for (let i = 0; i < imageFiles.length; i += batchSize) {
    const batch = imageFiles.slice(i, i + batchSize);
    updateLoadingProgress(
      Math.min(i + batchSize, imageFiles.length),
      imageFiles.length
    );

    const batchResults = await Promise.all(
      batch.map(async (file) => {
        const url = URL.createObjectURL(file);
        const dims = await getImageDimensions(url);
        return {
          url,
          name: file.name,
          size: formatFileSize(file.size),
          width: dims.width,
          height: dims.height,
          datasets: [],
          file,
        };
      })
    );

    allImages.push(...batchResults);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  showLoading(false);
  await showMainInterface();
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / (1024 * 1024)).toFixed(1) + "MB";
}

async function showMainInterface() {
  document.getElementById("initialScreen").classList.add("hidden");
  document.getElementById("mainInterface").classList.remove("hidden");
  await loadDatasetStructures();
  initDatasetButtons();
  await renderPage();
}

const updateMasonryLayout = debounce(() => {
  const grid = document.getElementById("imageGrid");
  if (masonry) masonry.destroy();
  masonry = new Masonry(grid, {
    itemSelector: ".masonry-grid-item",
    columnWidth: ".masonry-grid-item",
    percentPosition: true,
    gutter: 0,
  });
  imagesLoaded(grid, () => masonry.layout());
}, 100);

async function renderPage() {
  const start = (currentPage - 1) * imagesPerPage;
  const end = start + imagesPerPage;
  const pageImages = allImages.slice(start, end);

  const grid = document.getElementById("imageGrid");
  grid.innerHTML = ""; // Clear grid only

  const fragment = document.createDocumentFragment();
  const aspectRatio = targetHeight / targetWidth;

  pageImages.forEach((img, idx) => {
    const globalIdx = start + idx;
    const selectedIdx = selectedImages.findIndex((s) => s.index === globalIdx);

    // Create main container
    const item = document.createElement("div");
    item.className = "masonry-grid-item";
    item.dataset.index = globalIdx;

    if (selectedIdx === 0) item.classList.add("selected-target");
    else if (selectedIdx > 0) item.classList.add("selected-control");

    // Create placeholder div
    const placeholderDiv = document.createElement("div");
    placeholderDiv.className = "image-placeholder";
    placeholderDiv.style.paddingBottom = aspectRatio * 100 + "%";
    placeholderDiv.style.position = "relative";
    placeholderDiv.style.width = "100%";

    // Create img element with ALL attributes
    const imgElement = document.createElement("img");
    imgElement.setAttribute("data-src", img.url);
    imgElement.setAttribute("alt", img.name || "");
    imgElement.style.position = "absolute";
    imgElement.style.top = "0";
    imgElement.style.left = "0";
    imgElement.style.width = "100%";
    imgElement.style.height = "100%";
    imgElement.style.objectFit = "cover";
    imgElement.style.opacity = "0";
    imgElement.style.transition = "opacity 0.3s ease";

    // Append img to placeholder
    placeholderDiv.appendChild(imgElement);
    item.appendChild(placeholderDiv);

    // Create info labels
    const truncatedName = truncateFilename(img.name, 18);

    const infoTL = document.createElement("div");
    infoTL.className = "image-info info-tl";
    infoTL.title = img.name;
    infoTL.textContent = truncatedName;
    item.appendChild(infoTL);

    const needsScale = img.width > 0 && img.height > 0;
    const needsUpscale =
      needsScale && (img.width < targetWidth || img.height < targetHeight);
    const needsDownscale =
      needsScale && (img.width > targetWidth || img.height > targetHeight);

    const infoTR = document.createElement("div");
    infoTR.className = "image-info info-tr";
    if (needsUpscale) infoTR.classList.add("scale-up");
    else if (needsDownscale) infoTR.classList.add("scale-down");
    infoTR.textContent = img.width + "x" + img.height;
    item.appendChild(infoTR);

    if (selectedIdx >= 0) {
      const infoBL = document.createElement("div");
      infoBL.className = "image-info info-bl";
      if (selectedIdx > 0) infoBL.classList.add("control");
      infoBL.textContent =
        selectedIdx === 0 ? "Target" : "Control #" + selectedIdx;
      item.appendChild(infoBL);
    }

    if (img.datasets && img.datasets.length > 0) {
      const infoBR = document.createElement("div");
      infoBR.className = "image-info info-br";
      infoBR.textContent = "D: " + img.datasets.join(",");
      item.appendChild(infoBR);
    }

    // Add click handler
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      selectImage(globalIdx);
    });

    fragment.appendChild(item);
  });

  grid.appendChild(fragment);

  console.log("Created", grid.children.length, "items");
  if (grid.children[0]) {
    console.log("First img element:", grid.children[0].querySelector("img"));
  }

  // Initialize masonry
  updateMasonryLayout();

  // Load images
  loadImagesForCurrentPage(pageImages);

  updatePagination();
  updateImageCount();
  updateCompareButton();
}

async function loadImagesForCurrentPage(pageImages) {
  const grid = document.getElementById("imageGrid");

  for (let idx = 0; idx < pageImages.length; idx++) {
    const img = pageImages[idx];
    const item = grid.children[idx];

    if (!item) continue;

    const imgElement = item.querySelector("img[data-src]");
    const placeholderDiv = item.querySelector(".image-placeholder");

    if (!imgElement) continue;

    try {
      const preview = await getPreviewImage(img.url);

      // Arrow function preserves scope
      imgElement.onload = () => {
        imgElement.style.opacity = "1";
        if (placeholderDiv) {
          placeholderDiv.classList.add("loaded");
        }
      };

      imgElement.src = preview;
    } catch (error) {
      console.error("Load error:", error);
      imgElement.src = img.url;
      imgElement.style.opacity = "0.7";
      if (placeholderDiv) {
        placeholderDiv.classList.add("loaded");
      }
    }
  }
}

function updatePagination() {
  const totalPages = Math.ceil(allImages.length / imagesPerPage);
  const pageNumbers = document.getElementById("pageNumbers");
  pageNumbers.innerHTML = "";

  const maxButtons = 7;
  let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
  let endPage = Math.min(totalPages, startPage + maxButtons - 1);

  if (endPage - startPage < maxButtons - 1) {
    startPage = Math.max(1, endPage - maxButtons + 1);
  }

  for (let i = startPage; i <= endPage; i++) {
    const btn = document.createElement("button");
    btn.className = `page-btn ${i === currentPage ? "active" : ""}`;
    btn.textContent = i;
    btn.onclick = () => goToPage(i);
    pageNumbers.appendChild(btn);
  }

  document.getElementById("prevPageBtn").disabled = currentPage === 1;
  document.getElementById("nextPageBtn").disabled = currentPage === totalPages;
  document.getElementById(
    "pageInfo"
  ).textContent = `${currentPage} / ${totalPages}`;
}

function updateImageCount() {
  const count = `${allImages.length} images`;
  document.getElementById("imageCount").textContent = count;
  const mobileCount = document.getElementById("imageCountMobile");
  if (mobileCount) mobileCount.textContent = count;
}

function goToPage(page) {
  currentPage = page;
  renderPage();
}

function prevPage() {
  if (currentPage > 1) goToPage(currentPage - 1);
}

function nextPage() {
  const totalPages = Math.ceil(allImages.length / imagesPerPage);
  if (currentPage < totalPages) goToPage(currentPage + 1);
}

function selectImage(idx) {
  const existingIdx = selectedImages.findIndex((s) => s.index === idx);

  if (existingIdx >= 0) {
    selectedImages.splice(existingIdx, 1);
  } else {
    selectedImages.push({ index: idx, image: allImages[idx] });
  }

  updateSelectedStates();
  updateCompareButton();
}

function updateSelectedStates() {
  const items = document.querySelectorAll(".masonry-grid-item");
  items.forEach((item) => {
    const idx = parseInt(item.dataset.index);
    const selectedIdx = selectedImages.findIndex((s) => s.index === idx);

    item.classList.remove("selected-target", "selected-control");

    if (selectedIdx === 0) {
      item.classList.add("selected-target");
    } else if (selectedIdx > 0) {
      item.classList.add("selected-control");
    }

    const existingLabel = item.querySelector(".info-bl");
    if (existingLabel) existingLabel.remove();

    if (selectedIdx >= 0) {
      const label = document.createElement("div");
      label.className = `image-info info-bl ${
        selectedIdx > 0 ? "control" : ""
      }`;
      label.textContent =
        selectedIdx === 0 ? "Target" : `Control #${selectedIdx}`;
      item.appendChild(label);
    }
  });
}

function unselectAll() {
  selectedImages = [];
  updateSelectedStates();
  updateCompareButton();
}

function updateCompareButton() {
  document.getElementById("compareBtn").disabled = selectedImages.length < 2;
}

function openComparisonModal() {
  if (selectedImages.length < 2) return;

  document.body.classList.add("modal-open");
  document.getElementById("comparisonModal").classList.remove("hidden");
  currentControlIndex = 0;
  setComparisonMode(comparisonMode);
}

function closeComparisonModal() {
  document.body.classList.remove("modal-open");
  document.getElementById("comparisonModal").classList.add("hidden");
}

function setComparisonMode(mode) {
  comparisonMode = mode;

  ["modeSide", "modeOverlay", "modeBlend"].forEach((id) => {
    const btn = document.getElementById(id);
    const isActive =
      (mode === "side" && id === "modeSide") ||
      (mode === "overlay" && id === "modeOverlay") ||
      (mode === "blend" && id === "modeBlend");
    btn.className = `mode-btn ${isActive ? "active" : ""}`;
  });

  renderComparison();
}

async function renderComparison() {
  const container = document.getElementById("comparisonContainer");
  const controlsContainer = document.getElementById("comparisonControls");
  container.innerHTML = '<div class="loading-spinner"></div>';
  controlsContainer.innerHTML = "";

  const target = selectedImages[0].image;
  const controls = selectedImages.slice(1).map((s) => s.image);

  const targetConverted = await convertImageToTargetSize(
    target.url,
    targetWidth,
    targetHeight
  );
  const controlsConverted = await Promise.all(
    controls.map((c) =>
      convertImageToTargetSize(c.url, targetWidth, targetHeight)
    )
  );

  container.innerHTML = "";

  if (comparisonMode === "side") {
    renderSideBySide(container, targetConverted, controlsConverted);
  } else if (comparisonMode === "overlay") {
    renderOverlay(
      container,
      controlsContainer,
      targetConverted,
      controlsConverted
    );
  } else if (comparisonMode === "blend") {
    renderBlend(
      container,
      controlsContainer,
      targetConverted,
      controlsConverted
    );
  }
}

function renderSideBySide(container, target, controls) {
  const allImages = [target, ...controls];
  const maxHeight = window.innerHeight * 0.65;

  const scrollWrapper = document.createElement("div");
  scrollWrapper.className = "side-by-side-scroll";
  scrollWrapper.style.maxHeight = maxHeight + "px";

  const innerContainer = document.createElement("div");
  innerContainer.className = "side-by-side-container";

  allImages.forEach((imgData, idx) => {
    const div = document.createElement("div");
    div.className = "side-by-side-item";
    const img = document.createElement("img");
    img.src = imgData;
    img.style.maxHeight = maxHeight - 20 + "px";
    img.style.height = "auto";
    img.style.width = "auto";

    const label = document.createElement("div");
    label.className = "side-label";
    label.textContent = idx === 0 ? "Target" : `Control #${idx}`;

    div.appendChild(img);
    div.appendChild(label);
    innerContainer.appendChild(div);
  });

  scrollWrapper.appendChild(innerContainer);
  container.appendChild(scrollWrapper);
}

function renderOverlay(container, controlsContainer, target, controls) {
  currentControlIndex = Math.min(currentControlIndex, controls.length - 1);

  const wrapper = document.createElement("div");
  wrapper.className = "comparison-slider-wrapper";
  wrapper.style.position = "relative";
  wrapper.style.display = "inline-block";

  const img1 = document.createElement("img");
  img1.src = target;
  img1.style.display = "block";
  img1.style.maxHeight = "calc(75vh - 200px)";
  img1.style.maxWidth = "90vw";
  img1.style.height = "auto";
  img1.style.width = "auto";

  const clipContainer = document.createElement("div");
  clipContainer.style.position = "absolute";
  clipContainer.style.top = "0";
  clipContainer.style.left = "0";
  clipContainer.style.width = "50%";
  clipContainer.style.height = "100%";
  clipContainer.style.overflow = "hidden";

  const img2 = document.createElement("img");
  img2.src = controls[currentControlIndex];
  img2.style.display = "block";
  img2.style.maxHeight = "calc(75vh - 200px)";
  img2.style.maxWidth = "90vw";
  img2.style.height = "auto";
  img2.style.width = "auto";

  clipContainer.appendChild(img2);

  const handle = document.createElement("div");
  handle.className = "comparison-slider-handle";
  handle.style.left = "50%";

  wrapper.appendChild(img1);
  wrapper.appendChild(clipContainer);
  wrapper.appendChild(handle);
  container.appendChild(wrapper);

  let isActive = false;

  function updateSlider(clientX) {
    const rect = wrapper.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    clipContainer.style.width = percentage + "%";
    handle.style.left = percentage + "%";
  }

  handle.addEventListener("mousedown", (e) => {
    isActive = true;
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (isActive) updateSlider(e.clientX);
  });

  document.addEventListener("mouseup", () => (isActive = false));

  wrapper.addEventListener("click", (e) => {
    if (!isActive && e.target !== handle) updateSlider(e.clientX);
  });

  if (controls.length > 1) {
    const btnGroup = document.createElement("div");
    btnGroup.className = "flex gap-2";
    controls.forEach((c, i) => {
      const btn = document.createElement("button");
      btn.className = `control-selector-btn ${
        i === currentControlIndex ? "active" : ""
      }`;
      btn.textContent = `Control #${i + 1}`;
      btn.onclick = () => {
        currentControlIndex = i;
        renderComparison();
      };
      btnGroup.appendChild(btn);
    });
    controlsContainer.appendChild(btnGroup);
  }
}

function renderBlend(container, controlsContainer, target, controls) {
  currentControlIndex = Math.min(currentControlIndex, controls.length - 1);

  const wrapper = document.createElement("div");
  wrapper.style.position = "relative";
  wrapper.style.display = "inline-block";

  const img1 = document.createElement("img");
  img1.src = target;
  img1.style.display = "block";
  img1.style.maxHeight = "calc(75vh - 200px)";
  img1.style.maxWidth = "90vw";
  img1.style.height = "auto";
  img1.style.width = "auto";
  img1.style.position = "relative";

  const img2 = document.createElement("img");
  img2.src = controls[currentControlIndex];
  img2.id = "blendTopImage";
  img2.style.position = "absolute";
  img2.style.top = "0";
  img2.style.left = "0";
  img2.style.display = "block";
  img2.style.maxHeight = "calc(75vh - 200px)";
  img2.style.maxWidth = "90vw";
  img2.style.height = "auto";
  img2.style.width = "auto";
  img2.style.opacity = "0.5";

  wrapper.appendChild(img1);
  wrapper.appendChild(img2);
  container.appendChild(wrapper);

  const sliderDiv = document.createElement("div");
  sliderDiv.className = "mt-4";
  sliderDiv.style.width = "400px";
  sliderDiv.style.maxWidth = "90%";
  sliderDiv.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-400 font-semibold">Target</span>
                    <span class="text-xs text-gray-400 font-semibold">Control</span>
                </div>
                <input type="range" min="0" max="100" value="50" id="blendSlider">
            `;
  controlsContainer.appendChild(sliderDiv);

  setTimeout(() => {
    const slider = document.getElementById("blendSlider");
    if (slider) {
      slider.oninput = (e) => {
        const img = document.getElementById("blendTopImage");
        if (img) img.style.opacity = e.target.value / 100;
      };
    }
  }, 100);

  if (controls.length > 1) {
    const btnGroup = document.createElement("div");
    btnGroup.className = "flex gap-2 mt-3";
    controls.forEach((c, i) => {
      const btn = document.createElement("button");
      btn.className = `control-selector-btn ${
        i === currentControlIndex ? "active" : ""
      }`;
      btn.textContent = `Control #${i + 1}`;
      btn.onclick = () => {
        currentControlIndex = i;
        renderComparison();
      };
      btnGroup.appendChild(btn);
    });
    controlsContainer.appendChild(btnGroup);
  }
}

async function saveToDataset(datasetId) {
  if (selectedImages.length < 2) return;

  const controlCount = selectedImages.length - 1;

  if (datasets[datasetId].structure) {
    if (datasets[datasetId].structure.controls !== controlCount) {
      alert(
        `Dataset ${datasetId} requires 1 target + ${
          datasets[datasetId].structure.controls
        } controls.\nYou selected 1 target + ${controlCount} controls.\nPlease ${
          controlCount < datasets[datasetId].structure.controls
            ? "add"
            : "remove"
        } ${Math.abs(
          datasets[datasetId].structure.controls - controlCount
        )} control(s).`
      );
      return;
    }
  }

  scrollPosition = window.scrollY;
  showSavingIndicators(datasetId);

  const imagesData = await Promise.all(
    selectedImages.map(async (s) => {
      const converted = await convertImageToTargetSize(
        s.image.url,
        targetWidth,
        targetHeight
      );
      return { data: converted };
    })
  );

  const saveData = {
    datasetId,
    images: imagesData,
  };

  try {
    const response = await fetch("http://localhost:3000/api/dataset/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(saveData),
    });

    const result = await response.json();

    if (!response.ok) {
      if (result.expected !== undefined) {
        alert(
          `Dataset ${datasetId} requires 1 target + ${
            result.expected
          } controls.\nYou selected 1 target + ${
            result.received
          } controls.\nPlease ${
            result.received < result.expected ? "add" : "remove"
          } ${Math.abs(result.expected - result.received)} control(s).`
        );
      } else {
        alert("Error: " + result.message);
      }
      return;
    }

    if (result.success) {
      if (!datasets[datasetId].structure) {
        datasets[datasetId].structure = { target: 1, controls: controlCount };
      }
      if (!datasets[datasetId].active) {
        datasets[datasetId].active = true;
      }
      datasets[datasetId].count++;

      selectedImages.forEach((s) => {
        if (!allImages[s.index].datasets.includes(datasetId)) {
          allImages[s.index].datasets.push(datasetId);
        }
      });

      initDatasetButtons();
      closeComparisonModal();

      setTimeout(() => {
        const itemsToUpdate = [...selectedImages];
        selectedImages = [];
        updateCompareButton();

        itemsToUpdate.forEach((s) => {
          const item = document.querySelector(`[data-index="${s.index}"]`);
          if (item) {
            item.classList.remove("selected-target", "selected-control");
            const label = item.querySelector(".info-bl");
            if (label) label.remove();

            const existingBr = item.querySelector(".info-br");
            if (existingBr) existingBr.remove();

            if (allImages[s.index].datasets.length > 0) {
              const br = document.createElement("div");
              br.className = "image-info info-br";
              br.textContent = `D: ${allImages[s.index].datasets.join(",")}`;
              item.appendChild(br);
            }
          }
        });

        window.scrollTo(0, scrollPosition);
      }, 100);
    }
  } catch (error) {
    console.error("Save error:", error);
    alert("Error saving dataset: " + error.message);
  }
}

function showSavingIndicators(datasetId) {
  selectedImages.forEach((s) => {
    const start = (currentPage - 1) * imagesPerPage;
    const pageIdx = s.index - start;

    if (pageIdx >= 0 && pageIdx < imagesPerPage) {
      const item = document.querySelector(`[data-index="${s.index}"]`);
      if (item) {
        const indicator = document.createElement("div");
        indicator.className = "saving-indicator";
        indicator.textContent = `✓ Saved to Dataset ${datasetId}`;
        item.appendChild(indicator);

        setTimeout(() => indicator.remove(), 1200);
      }
    }
  });
}

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  if (e.key === "c" || e.key === "C") {
    if (selectedImages.length >= 2) openComparisonModal();
  } else if (e.key === "u" || e.key === "U") {
    closeComparisonModal();
    unselectAll();
  } else if (e.key === "Escape") {
    closeComparisonModal();
  } else if (e.key >= "0" && e.key <= "9") {
    const datasetId = parseInt(e.key);
    if (selectedImages.length >= 2) {
      saveToDataset(datasetId);
    }
  }
});

initDatasetButtons();
