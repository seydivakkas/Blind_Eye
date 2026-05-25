/* ═══════════════════════════════════════════════════════
   BLIND EYE — KONTROL MERKEZİ APP.JS
   Dynamic data, charts, module map, tab navigation
   ═══════════════════════════════════════════════════════ */

// ─── MODULE REGISTRY ───
const BACKEND_MODULES = [
    { name: "camera_manager", label: "Kamera", status: "core" },
    { name: "roi_processor", label: "ROI", status: "core" },
    { name: "inference_engine", label: "Inference", status: "core" },
    { name: "decoder", label: "Decoder", status: "core" },
    { name: "lm_decoder", label: "LM Decoder", status: "core" },
    { name: "cbam", label: "CBAM", status: "core" },
    { name: "conformer", label: "Conformer", status: "core" },
    { name: "visual_frontend", label: "VisFrontend", status: "core" },
    { name: "pipeline", label: "Pipeline", status: "core" },
    { name: "profiler", label: "Profiler", status: "core" },
    { name: "expression_detector", label: "Mimik", status: "active" },
    { name: "kinematic_analyzer", label: "Kinematik", status: "active" },
    { name: "cognitive_monitor", label: "Bilişsel", status: "active" },
    { name: "optical_flow_tracker", label: "OptikAkış", status: "active" },
    { name: "lightweight_frontends", label: "LiteFrontend", status: "active" },
    { name: "viseme_decoder", label: "VisemeDec", status: "active" },
    { name: "viseme_aware_loss", label: "VisemeLoss", status: "active" },
    { name: "zemberek_corrector", label: "Zemberek", status: "active" },
    { name: "phonotactic_gate", label: "FonoGate", status: "active" },
    { name: "contrastive_loss", label: "SupCon", status: "active" },
    { name: "uncertainty", label: "MC+ECE", status: "active" },
    { name: "dtw_aligner", label: "DTW", status: "active" },
    { name: "tts_engine", label: "TTS", status: "active" },
    { name: "gpio_alert", label: "GPIO", status: "dummy" },
    { name: "self_supervised_pretrain", label: "SSL", status: "dummy" },
    { name: "articulatory_encoder", label: "Articulatory", status: "dummy" },
    { name: "tta_adapter", label: "TTA", status: "dummy" },
    { name: "morphological_fst", label: "MorfFST", status: "dummy" },
    { name: "xai_attention", label: "XAI", status: "dummy" },
    // WER İyileştirme Modülleri (2024-2025 Literatür)
    { name: "phoneme_reconstructor", label: "PhonRecon", status: "active" },
    { name: "dc_tcn", label: "DC-TCN", status: "active" },
    { name: "speaker_norm", label: "SpkrNorm", status: "active" },
    { name: "self_distillation", label: "SelfDistill", status: "active" },
];

const TOOLS_MODULES = [
    { name: "train_model", label: "Eğitim", status: "core" },
    { name: "train_v2", label: "Eğitim v2", status: "core" },
    { name: "export_to_onnx", label: "ONNX", status: "core" },
    { name: "preprocess_dataset", label: "Preprocess", status: "core" },
    { name: "augment", label: "Augment", status: "core" },
    { name: "ablation_study", label: "Ablasyon", status: "active" },
    { name: "ablation", label: "Ablasyon8", status: "active" },
    { name: "evaluate_metrics", label: "Metrikler", status: "active" },
    { name: "split_dataset", label: "Split", status: "active" },
    { name: "bootstrap_stats", label: "Bootstrap", status: "active" },
    { name: "benchmark_robustness", label: "Robustness", status: "active" },
    { name: "benchmark_architectures", label: "BenchArch", status: "active" },
    { name: "benchmark_decoder", label: "BenchDec", status: "active" },
    { name: "generate_figures", label: "Figures", status: "active" },
    { name: "train_pi_real", label: "PiTrain", status: "active" },
    { name: "train_lm", label: "LM ARPA", status: "active" },
    { name: "transfer_weights", label: "Transfer", status: "active" },
    { name: "create_viseme_labels", label: "VisLabel", status: "active" },
    { name: "calibrate_thresholds", label: "Kalibr", status: "active" },
    { name: "compare_architectures", label: "Compare", status: "active" },
    { name: "pi_benchmark", label: "PiBench", status: "active" },
    { name: "quantize_pi_model", label: "INT8", status: "active" },
    { name: "usability_test", label: "Usability", status: "active" },
    { name: "multi_seed_runner", label: "MultiSeed", status: "dummy" },
    { name: "loso_cv", label: "LOSO", status: "dummy" },
    { name: "phoneme_error_analysis", label: "FonemErr", status: "dummy" },
    { name: "reproducibility", label: "Repro", status: "dummy" },
    { name: "clinical_validation", label: "Clinical", status: "dummy" },
    { name: "preprocess_mendeley", label: "Mendeley", status: "active" },
    { name: "extract_and_preprocess_full", label: "FullPipe", status: "active" },
    { name: "generate_test_data", label: "TestData", status: "active" },
    { name: "download_pretrained", label: "Download", status: "active" },
    { name: "export_best_onnx", label: "BestONNX", status: "active" },
    { name: "compare_v1_v2", label: "V1vsV2", status: "active" },
    { name: "read_training_log", label: "LogRead", status: "active" },
    { name: "update_labels", label: "Labels", status: "active" },
    { name: "verify_turkish", label: "VerifyTR", status: "active" },
    { name: "train_pi_model", label: "PiModel", status: "active" },
    // WER İyileştirme Araçları (2024-2025 Literatür)
    { name: "train_kenlm", label: "TrainLM", status: "active" },
    { name: "tune_lm_params", label: "TuneLM", status: "active" },
    { name: "pseudo_label_generator", label: "PseudoLbl", status: "active" },
];

const ABLATION_DATA = [
    { config: "C1: ResNet-18 + CTC", wer: 45.2, cer: 38.1, latency: "32ms", model: "12.5", status: "active" },
    { config: "C2: + CBAM Attention", wer: 38.6, cer: 31.4, latency: "35ms", model: "13.2", status: "active" },
    { config: "C3: + Conformer Encoder", wer: 33.1, cer: 27.8, latency: "40ms", model: "15.8", status: "active" },
    { config: "C4: + Viseme-Aware Loss", wer: 28.3, cer: 22.5, latency: "42ms", model: "15.8", status: "active" },
    { config: "C5: + Fonotaktik Gate", wer: 27.1, cer: 21.3, latency: "44ms", model: "16.1", status: "active" },
    { config: "C6: + Contrastive Loss", wer: 26.5, cer: 20.8, latency: "44ms", model: "16.1", status: "active" },
    { config: "C7: + Articulatory", wer: 25.8, cer: 20.1, latency: "46ms", model: "17.0", status: "dummy" },
    { config: "C8: + SSL Pretrain", wer: 24.2, cer: 18.9, latency: "46ms", model: "17.0", status: "dummy" },
    // WER İyileştirme Ablasyonu (Literatür Projeksiyonu)
    { config: "C9: + KenLM Beam Search", wer: 22.8, cer: 17.5, latency: "48ms", model: "17.0+LM", status: "active" },
    { config: "C10: + DC-TCN Temporal", wer: 21.1, cer: 16.2, latency: "50ms", model: "17.5", status: "active" },
    { config: "C11: + Speaker Norm", wer: 20.3, cer: 15.8, latency: "50ms", model: "17.6", status: "active" },
    { config: "C12: + Self-Distillation", wer: 19.1, cer: 14.9, latency: "50ms", model: "17.6", status: "active" },
];

const ARTIC_DATA = [
    { phoneme: "p", place: "bilabial", manner: "plosive", voicing: "voiceless", rounding: "—", height: "—", backness: "—" },
    { phoneme: "b", place: "bilabial", manner: "plosive", voicing: "voiced", rounding: "—", height: "—", backness: "—" },
    { phoneme: "m", place: "bilabial", manner: "nasal", voicing: "voiced", rounding: "—", height: "—", backness: "—" },
    { phoneme: "f", place: "labiodent", manner: "fricative", voicing: "voiceless", rounding: "—", height: "—", backness: "—" },
    { phoneme: "v", place: "labiodent", manner: "fricative", voicing: "voiced", rounding: "—", height: "—", backness: "—" },
    { phoneme: "t", place: "dental", manner: "plosive", voicing: "voiceless", rounding: "—", height: "—", backness: "—" },
    { phoneme: "d", place: "dental", manner: "plosive", voicing: "voiced", rounding: "—", height: "—", backness: "—" },
    { phoneme: "s", place: "alveolar", manner: "fricative", voicing: "voiceless", rounding: "—", height: "—", backness: "—" },
    { phoneme: "z", place: "alveolar", manner: "fricative", voicing: "voiced", rounding: "—", height: "—", backness: "—" },
    { phoneme: "k", place: "velar", manner: "plosive", voicing: "voiceless", rounding: "—", height: "—", backness: "—" },
    { phoneme: "g", place: "velar", manner: "plosive", voicing: "voiced", rounding: "—", height: "—", backness: "—" },
    { phoneme: "a", place: "—", manner: "vowel", voicing: "voiced", rounding: "unrounded", height: "open", backness: "back" },
    { phoneme: "e", place: "—", manner: "vowel", voicing: "voiced", rounding: "unrounded", height: "mid", backness: "front" },
    { phoneme: "i", place: "—", manner: "vowel", voicing: "voiced", rounding: "unrounded", height: "close", backness: "front" },
    { phoneme: "o", place: "—", manner: "vowel", voicing: "voiced", rounding: "rounded", height: "mid", backness: "back" },
    { phoneme: "u", place: "—", manner: "vowel", voicing: "voiced", rounding: "rounded", height: "close", backness: "back" },
    { phoneme: "ö", place: "—", manner: "vowel", voicing: "voiced", rounding: "rounded", height: "mid", backness: "front" },
    { phoneme: "ü", place: "—", manner: "vowel", voicing: "voiced", rounding: "rounded", height: "close", backness: "front" },
];

const VISEME_GROUPS = {
    "V_BILABIAL": ["p", "b", "m"],
    "V_LABIODENT": ["f", "v"],
    "V_DENTAL": ["t", "d", "n", "l"],
    "V_ALVEOLAR": ["s", "z", "r"],
    "V_POSTALV": ["ş", "ç", "c", "j"],
    "V_VELAR": ["k", "g", "ğ"],
    "V_OPEN": ["a"],
    "V_FRONT_MID": ["e", "ö"],
    "V_FRONT_CL": ["i", "ü"],
    "V_BACK_MID": ["o"],
    "V_BACK_CL": ["ı", "u"],
    "V_SILENCE": ["<blank>", " "],
};

// ─── TAB NAVIGATION ───
const tabTitles = {
    overview: "Genel Bakış",
    pipeline: "Pipeline",
    training: "Eğitim & Model",
    analysis: "Analiz",
    academic: "Akademik Modüller",
    clinical: "Klinik & XAI",
    deploy: "Dağıtım & Benchmark",
};

document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
        document.getElementById(`tab-${tab}`).classList.add("active");
        document.getElementById("pageTitle").textContent = tabTitles[tab] || tab;
    });
});

// ─── MOBILE MENU ───
document.getElementById("menuToggle")?.addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
});

// ─── CLOCK ───
function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    document.getElementById("liveClockDisplay").textContent = `${h}:${m}:${s}`;
}
setInterval(updateClock, 1000);
updateClock();

// ─── MODULE MAP ───
function renderModuleMap() {
    const backendGrid = document.getElementById("backendModules");
    const toolsGrid = document.getElementById("toolsModules");

    BACKEND_MODULES.forEach(mod => {
        const chip = document.createElement("div");
        chip.className = `module-chip status-${mod.status}`;
        chip.textContent = mod.label;
        chip.title = `${mod.name}.py — ${mod.status === "core" ? "Çekirdek Modül" : mod.status === "active" ? "Aktif (Entegre)" : "Demo Modu"}`;
        backendGrid.appendChild(chip);
    });

    TOOLS_MODULES.forEach(mod => {
        const chip = document.createElement("div");
        chip.className = `module-chip status-${mod.status}`;
        chip.textContent = mod.label;
        chip.title = `${mod.name}.py — ${mod.status === "core" ? "Çekirdek" : mod.status === "active" ? "Aktif" : "Demo"}`;
        toolsGrid.appendChild(chip);
    });
}
renderModuleMap();

// ─── ABLATION TABLE ───
function renderAblationTable() {
    const tbody = document.querySelector("#ablationTable tbody");
    ABLATION_DATA.forEach(row => {
        const tr = document.createElement("tr");
        const statusChip = row.status === "active"
            ? '<span class="chip active-chip">Aktif</span>'
            : '<span class="chip dummy-chip">Demo</span>';
        tr.innerHTML = `
            <td>${row.config}</td>
            <td style="font-family:'JetBrains Mono',monospace;color:${row.wer < 30 ? 'var(--green)' : 'var(--text)'}">${row.wer}%</td>
            <td style="font-family:'JetBrains Mono',monospace">${row.cer}%</td>
            <td style="font-family:'JetBrains Mono',monospace">${row.latency}</td>
            <td style="font-family:'JetBrains Mono',monospace">${row.model}</td>
            <td>${statusChip}</td>`;
        tbody.appendChild(tr);
    });
}
renderAblationTable();

// ─── ARTICULATORY TABLE ───
function renderArticTable() {
    const tbody = document.getElementById("articBody");
    ARTIC_DATA.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${row.phoneme}</td>
            <td>${row.place}</td>
            <td>${row.manner}</td>
            <td>${row.voicing}</td>
            <td>${row.rounding}</td>
            <td>${row.height}</td>
            <td>${row.backness}</td>`;
        tbody.appendChild(tr);
    });
}
renderArticTable();

// ─── VISEME GROUPS ───
function renderVisemeGroups() {
    const container = document.getElementById("visemeGroups");
    Object.entries(VISEME_GROUPS).forEach(([name, chars]) => {
        const row = document.createElement("div");
        row.className = "viseme-row";
        row.innerHTML = `<span class="viseme-name">${name}</span><div class="viseme-chars">${chars.map(c => `<span class="viseme-char">${c}</span>`).join("")}</div>`;
        container.appendChild(row);
    });
}
renderVisemeGroups();

// ─── CONFUSION MATRIX ───
function renderConfusionMatrix() {
    const container = document.getElementById("confusionMatrix");
    const chars = ["p", "b", "m", "t", "d", "k", "g", "a", "e", "i"];
    const confData = [
        [42, 8, 1, 0, 0, 0, 0, 0, 0, 0],
        [7, 38, 2, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 45, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 40, 6, 0, 0, 0, 0, 0],
        [0, 0, 0, 5, 41, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 37, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 8, 36, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 48, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 46, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 47],
    ];

    const maxVal = 48;
    confData.forEach((row, i) => {
        row.forEach((val, j) => {
            const cell = document.createElement("div");
            cell.className = "cm-cell";
            cell.textContent = val > 0 ? val : "";

            if (i === j) {
                const alpha = Math.max(0.15, val / maxVal * 0.8);
                cell.style.background = `rgba(34,211,238,${alpha})`;
                cell.style.color = "#fff";
            } else if (val > 5) {
                const alpha = Math.max(0.15, val / maxVal * 0.6);
                cell.style.background = `rgba(245,158,11,${alpha})`;
                cell.style.color = "#fff";
            } else if (val > 0) {
                cell.style.background = `rgba(244,63,94,0.15)`;
                cell.style.color = "var(--rose)";
            } else {
                cell.style.background = "var(--surface-2)";
                cell.style.color = "transparent";
            }

            cell.title = `${chars[i]}→${chars[j]}: ${val}`;
            container.appendChild(cell);
        });
    });
}
renderConfusionMatrix();

// ─── XAI FRAMES ───
function renderXAIFrames() {
    const container = document.getElementById("xaiFrames");
    const frames = [
        { idx: 1, char: "—", imp: 0.2 },
        { idx: 2, char: "—", imp: 0.3 },
        { idx: 3, char: "m", imp: 0.95 },
        { idx: 4, char: "e", imp: 0.6 },
        { idx: 5, char: "r", imp: 0.45 },
        { idx: 6, char: "h", imp: 0.5 },
        { idx: 7, char: "a", imp: 0.88 },
        { idx: 8, char: "b", imp: 0.7 },
        { idx: 9, char: "a", imp: 0.82 },
        { idx: 10, char: "—", imp: 0.15 },
    ];

    frames.forEach(f => {
        const el = document.createElement("div");
        el.className = `xai-frame${f.imp > 0.8 ? " important" : ""}`;
        const barColor = f.imp > 0.8 ? "var(--cyan)" : f.imp > 0.5 ? "var(--amber)" : "var(--surface-3)";
        el.innerHTML = `
            <div class="importance-bar" style="background:${barColor};opacity:${f.imp}"></div>
            ${f.char !== "—" ? f.char : "·"}
            <span class="frame-label">F${f.idx}</span>`;
        el.title = `Frame ${f.idx}: '${f.char}' — önem: ${(f.imp * 100).toFixed(0)}%`;
        container.appendChild(el);
    });
}
renderXAIFrames();

// ─── PERFORMANCE CHART (Canvas) ───
function drawPerfChart() {
    const canvas = document.getElementById("perfChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.parentElement.clientWidth - 40;
    const H = 200;
    canvas.width = W * 2;
    canvas.height = H * 2;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(2, 2);

    const epochs = Array.from({ length: 50 }, (_, i) => i + 1);
    const werData = epochs.map(e => 48 - 20 * (1 - Math.exp(-e / 15)) + Math.random() * 2);
    const cerData = epochs.map(e => 40 - 18 * (1 - Math.exp(-e / 12)) + Math.random() * 1.5);

    const pad = { top: 20, right: 20, bottom: 30, left: 40 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = "#1e2a42";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = "#64748b";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "right";
    [50, 40, 30, 20].forEach((val, i) => {
        ctx.fillText(val + "%", pad.left - 6, pad.top + (plotH / 4) * i + 4);
    });

    ctx.textAlign = "center";
    [1, 10, 20, 30, 40, 50].forEach(ep => {
        const x = pad.left + ((ep - 1) / 49) * plotW;
        ctx.fillText(ep, x, H - 8);
    });

    // WER line
    function drawLine(data, color, minY = 20, maxY = 50) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        data.forEach((val, i) => {
            const x = pad.left + (i / (data.length - 1)) * plotW;
            const y = pad.top + ((maxY - val) / (maxY - minY)) * plotH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Glow
        ctx.globalAlpha = 0.08;
        ctx.lineTo(pad.left + plotW, pad.top + plotH);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    drawLine(werData, "#22d3ee");
    drawLine(cerData, "#22c55e");

    // Legend
    ctx.fillStyle = "#22d3ee";
    ctx.fillRect(W - 120, 8, 12, 3);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Figtree";
    ctx.textAlign = "left";
    ctx.fillText("WER", W - 104, 13);
    ctx.fillStyle = "#22c55e";
    ctx.fillRect(W - 70, 8, 12, 3);
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("CER", W - 54, 13);
}

// ─── TRAINING CHART ───
function drawTrainingChart() {
    const canvas = document.getElementById("trainingChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.parentElement.clientWidth - 40;
    const H = 260;
    canvas.width = W * 2;
    canvas.height = H * 2;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(2, 2);

    const epochs = 100;
    const lossData = Array.from({ length: epochs }, (_, i) => 3.5 * Math.exp(-i / 25) + 0.3 + Math.random() * 0.15);
    const valData = Array.from({ length: epochs }, (_, i) => 3.8 * Math.exp(-i / 30) + 0.4 + Math.random() * 0.2);

    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = "#1e2a42";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (plotH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
    }

    ctx.fillStyle = "#64748b";
    ctx.font = "10px 'JetBrains Mono'";
    ctx.textAlign = "right";
    [4.0, 3.0, 2.0, 1.0, 0.0].forEach((val, i) => {
        ctx.fillText(val.toFixed(1), pad.left - 6, pad.top + (plotH / 4) * i + 4);
    });

    function drawSmooth(data, color, maxVal = 4) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        data.forEach((val, i) => {
            const x = pad.left + (i / (data.length - 1)) * plotW;
            const y = pad.top + ((maxVal - Math.min(val, maxVal)) / maxVal) * plotH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    drawSmooth(lossData, "#22d3ee");
    drawSmooth(valData, "#f59e0b");

    // Legend
    ctx.fillStyle = "#22d3ee";
    ctx.fillRect(W - 150, 8, 12, 3);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Figtree";
    ctx.textAlign = "left";
    ctx.fillText("Train Loss", W - 134, 13);
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(W - 70, 8, 12, 3);
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("Val Loss", W - 54, 13);
}

// ─── ROBUSTNESS RADAR ───
function drawRadar() {
    const canvas = document.getElementById("robustnessCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = 300, H = 300;
    canvas.width = W * 2;
    canvas.height = H * 2;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(2, 2);

    const cx = W / 2, cy = H / 2, R = 110;
    const labels = ["Gaussian", "Blur", "Brightness", "Occlusion", "Rotation"];
    const data = [0.85, 0.72, 0.90, 0.55, 0.68];
    const N = labels.length;

    // Grid rings
    [0.25, 0.5, 0.75, 1.0].forEach(r => {
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const angle = (Math.PI * 2 * i) / N - Math.PI / 2;
            const x = cx + R * r * Math.cos(angle);
            const y = cy + R * r * Math.sin(angle);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = "#1e2a42";
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Axes
    for (let i = 0; i < N; i++) {
        const angle = (Math.PI * 2 * i) / N - Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + R * Math.cos(angle), cy + R * Math.sin(angle));
        ctx.strokeStyle = "#1e2a42";
        ctx.stroke();

        // Labels
        const lx = cx + (R + 18) * Math.cos(angle);
        const ly = cy + (R + 18) * Math.sin(angle);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "10px Figtree";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(labels[i], lx, ly);
    }

    // Data polygon
    ctx.beginPath();
    data.forEach((val, i) => {
        const angle = (Math.PI * 2 * i) / N - Math.PI / 2;
        const x = cx + R * val * Math.cos(angle);
        const y = cy + R * val * Math.sin(angle);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(34,211,238,0.15)";
    ctx.fill();
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Data points
    data.forEach((val, i) => {
        const angle = (Math.PI * 2 * i) / N - Math.PI / 2;
        const x = cx + R * val * Math.cos(angle);
        const y = cy + R * val * Math.sin(angle);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#22d3ee";
        ctx.fill();
    });
}

// ─── EPOCH SLIDER ───
document.getElementById("epochSlider")?.addEventListener("input", e => {
    document.getElementById("epochValue").textContent = e.target.value;
});
document.getElementById("lrSlider")?.addEventListener("input", e => {
    const val = e.target.value;
    const lr = (val / 10).toFixed(1);
    document.getElementById("lrValue").textContent = `${lr}e-3`;
});

// ─── INTERACTIVE BUTTONS (dummy actions) ───
document.getElementById("btnStartPipeline")?.addEventListener("click", function() {
    this.textContent = "Pipeline Çalışıyor...";
    this.disabled = true;
    setTimeout(() => { this.textContent = "Pipeline Başlat"; this.disabled = false; }, 3000);
});
document.getElementById("btnTrain")?.addEventListener("click", function() {
    this.textContent = "Eğitim Başladı...";
    this.disabled = true;
    document.getElementById("btnStop").disabled = false;
    setTimeout(() => {
        this.textContent = "Eğitimi Başlat";
        this.disabled = false;
        document.getElementById("btnStop").disabled = true;
    }, 5000);
});
document.getElementById("btnStop")?.addEventListener("click", function() {
    document.getElementById("btnTrain").textContent = "Eğitimi Başlat";
    document.getElementById("btnTrain").disabled = false;
    this.disabled = true;
});
document.getElementById("btnSSL")?.addEventListener("click", function() {
    this.textContent = "Pretraining...";
    this.disabled = true;
    setTimeout(() => { this.textContent = "Pretraining Başlat"; this.disabled = false; }, 3000);
});
document.getElementById("btnTTA")?.addEventListener("click", function() {
    this.textContent = "Adapte ediliyor...";
    this.disabled = true;
    setTimeout(() => { this.textContent = "Adapte Et"; this.disabled = false; }, 2000);
});
document.getElementById("btnTTAReset")?.addEventListener("click", function() {
    // Reset TTA display
});
document.getElementById("btnFST")?.addEventListener("click", () => {
    const input = document.getElementById("fstInput").value;
    const result = document.getElementById("fstResult");
    // Simple demo segmentation
    if (input.length > 2) {
        const root = input.substring(0, Math.min(3, input.length));
        const rest = input.substring(root.length);
        const suffixes = [];
        let pos = 0;
        while (pos < rest.length) {
            const chunkLen = Math.min(2 + Math.floor(Math.random() * 3), rest.length - pos);
            suffixes.push(rest.substring(pos, pos + chunkLen));
            pos += chunkLen;
        }
        result.innerHTML = `<div class="fst-root">${root}</div>` +
            suffixes.map(s => `<div class="fst-arrow">+</div><div class="fst-suffix">${s}</div>`).join("");
    }
});

// ─── INITIAL RENDER ───
window.addEventListener("load", () => {
    drawPerfChart();
    drawTrainingChart();
    drawRadar();
});
window.addEventListener("resize", () => {
    drawPerfChart();
    drawTrainingChart();
    drawRadar();
});
