/* ============================================================
   BIOACTIVE MOLECULE PREDICTOR - script.js
   Vanilla JavaScript - connects frontend to FastAPI backend
   ============================================================ */

// --- Dynamic Backend URL ---
const API_BASE = (window.location.protocol === "file:")
  ? "http://127.0.0.1:8000"
  : window.location.origin;

// --- Example SMILES to cycle through ---
const EXAMPLES = [
  "CCO",
  "CC(=O)Oc1ccccc1C(=O)O",
  "c1ccccc1",
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "CS(=O)(=O)c1ccc(-c2csc(CC(=O)O)c2-c2ccc(F)cc2)cc1"
];
let exampleIndex = 0;

// --- DOM Element References ---
const smilesInput    = document.getElementById("smilesInput");
const predictBtn     = document.getElementById("predictBtn");
const clearBtn       = document.getElementById("clearBtn");
const loadExBtn      = document.getElementById("loadExampleBtn");
const inputError     = document.getElementById("inputError");
const resultEmpty    = document.getElementById("resultEmpty");
const resultLoading  = document.getElementById("resultLoading");
const resultData     = document.getElementById("resultData");
const backendStatus  = document.getElementById("backendStatus");
const navToggle      = document.getElementById("navToggle");
const navMobile      = document.getElementById("navMobile");

// --- 1. Navigation ---
if (navToggle) {
  navToggle.addEventListener("click", () => {
    const isOpen = navMobile.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });
}

document.querySelectorAll(".mobile-link").forEach((link) => {
  link.addEventListener("click", () => {
    navMobile.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  });
});

document.addEventListener("click", (e) => {
  if (navMobile && navToggle && !navMobile.contains(e.target) && !navToggle.contains(e.target)) {
    navMobile.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  }
});

document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth" });
    }
  });
});

// --- 2. Backend Health Check ---
async function checkBackendHealth() {
  if (!backendStatus) return;
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      backendStatus.textContent = "Online";
      backendStatus.style.color = "#16a34a";
    } else {
      backendStatus.textContent = "Server response error";
      backendStatus.style.color = "#d97706";
    }
  } catch {
    backendStatus.textContent = "Offline (run: uvicorn main:app --reload --port 8000)";
    backendStatus.style.color = "#dc2626";
  }
}

checkBackendHealth();

// --- 3. Load Example ---
function loadExample() {
  smilesInput.value = EXAMPLES[exampleIndex % EXAMPLES.length];
  exampleIndex++;
  clearError();
  smilesInput.focus();
}

if (loadExBtn) {
  loadExBtn.addEventListener("click", loadExample);
}

// --- 4. Input Validation ---
function validateInput() {
  const raw = smilesInput.value.trim();
  if (raw === "") {
    showError("Please enter a SMILES string before predicting.");
    return false;
  }
  if (/^\d+$/.test(raw)) {
    showError("This doesn't look like a valid SMILES string. Example: CCO");
    return false;
  }
  if (/\s/.test(raw)) {
    showError("SMILES strings should not contain spaces. Please check your input.");
    return false;
  }
  clearError();
  return true;
}

function showError(msg) {
  inputError.textContent = msg;
  inputError.classList.add("visible");
  smilesInput.classList.add("input-error");
}

function clearError() {
  inputError.textContent = "";
  inputError.classList.remove("visible");
  smilesInput.classList.remove("input-error");
}

if (smilesInput) {
  smilesInput.addEventListener("input", clearError);
}

// --- 5. Loading State ---
function setLoadingState(isLoading) {
  if (isLoading) {
    resultEmpty.style.display   = "none";
    resultLoading.style.display = "flex";
    resultData.style.display    = "none";
    predictBtn.disabled         = true;
    predictBtn.textContent      = "Predicting...";
  } else {
    resultLoading.style.display = "none";
    predictBtn.disabled         = false;
    predictBtn.textContent      = "Predict Bioactivity";
  }
}

// --- 6. Display Result ---
function displayPrediction(data) {
  const isActive    = String(data.prediction).toLowerCase() === "active";
  const badgeClass  = isActive ? "badge-active" : "badge-inactive";
  const label       = isActive ? "Active" : "Inactive";
  const note        = isActive ? "(Biologically Active)" : "(Biologically Inactive)";
  const probPercent = (data.probability * 100).toFixed(1) + "%";

  const displaySmiles = data.smiles.length > 40
    ? data.smiles.substring(0, 40) + "..."
    : data.smiles;

  resultData.innerHTML = `
    <div class="result-badge ${badgeClass}">
      ${label}
      <span style="font-size:0.85rem; font-weight:500; opacity:0.85;">${note}</span>
    </div>
    <div class="result-rows">
      <div class="result-row">
        <span class="result-row-label">Prediction</span>
        <span class="result-row-value">${label}</span>
      </div>
      <div class="result-row">
        <span class="result-row-label">Confidence Probability</span>
        <span class="result-row-value">${probPercent}</span>
      </div>
      <div class="result-row">
        <span class="result-row-label">Model Used</span>
        <span class="result-row-value">${data.model}</span>
      </div>
      <div class="result-row">
        <span class="result-row-label">SMILES Input</span>
        <span class="result-row-value" style="font-family:monospace; font-size:0.8rem;">${displaySmiles}</span>
      </div>
    </div>
  `;

  resultData.style.display = "block";
}

// --- 7. Prediction API Call ---
async function runPrediction() {
  if (!validateInput()) return;
  const smiles = smilesInput.value.trim();

  setLoadingState(true);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Unknown server error" }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    displayPrediction(data);
  } catch (err) {
    showError("Prediction failed: " + err.message);
    resultEmpty.style.display = "flex";
  } finally {
    setLoadingState(false);
  }
}

if (predictBtn) {
  predictBtn.addEventListener("click", runPrediction);
}

if (smilesInput) {
  smilesInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runPrediction();
    }
  });
}

if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    smilesInput.value = "";
    clearError();
    resultEmpty.style.display = "flex";
    resultData.style.display  = "none";
    smilesInput.focus();
  });
}

// ============================================================
// DIAGNOSTICS STUDIO - Tab switching + Charts
// ============================================================

// --- Tab Switching ---
const studioTabBtns = document.querySelectorAll(".studio-tab-btn");
const tabPanels     = document.querySelectorAll(".tab-content-panel");

function switchTab(targetId) {
  studioTabBtns.forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === targetId);
  });
  tabPanels.forEach(panel => {
    panel.classList.toggle("active", panel.id === targetId);
  });

  // Lazy-init charts on first activation
  if (targetId === "tab-roc" && !rocChartInstance)           initRocChart();
  if (targetId === "tab-benchmark" && !benchmarkChartInstance) initBenchmarkChart();
  if (targetId === "tab-radar" && !radarChartInstance)       initRadarChart();
}

studioTabBtns.forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
});

// --- Page Ready ---
document.addEventListener("DOMContentLoaded", () => {
  // All interactive diagnostics removed - using static benchmark image.
  console.log("Bioactive Molecule Predictor frontend loaded.");
});
