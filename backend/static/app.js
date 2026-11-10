// Aurélia Speech Analytics — vanilla frontend.
// Screens toggled via display, no SPA framework.

(function () {
  "use strict";

  // ---------- Global state ----------
  const state = {
    presets: [],
    selectedPresets: new Set(["sentiment", "intent", "topics", "summary", "next_action"]),
    customQuestions: [""],
    transcript: "",
    recording: false,
    ws: null,
    recorder: null,
    stream: null,
    timerInterval: null,
    startTs: 0,
  };

  // ---------- Refs ----------
  const $ = (id) => document.getElementById(id);
  const pageDashboard = $("page-dashboard");
  const pageUpload = $("page-upload");
  const pageConfig = $("page-config");
  const pageRecord = $("page-record");
  const pageArchitecture = $("page-architecture");
  const pageFila = $("page-fila");
  const navTabs = document.querySelectorAll(".nav-tab");
  const presetsGrid = $("presets-grid");
  const questionsList = $("questions-list");
  const continueBtn = $("continue-btn");
  const addQuestionBtn = $("add-question-btn");
  const configError = $("config-error");
  const recError = $("record-error");
  const recButton = $("rec-button");
  const recStatus = $("rec-status");
  const timer = $("timer");
  const transcriptBox = $("transcript-box");
  const analysisEmpty = $("analysis-empty");
  const analysisLoading = $("analysis-loading");
  const analysisResult = $("analysis-result");
  const analysisError = $("analysis-error");
  const analyzeBtn = $("analyze-btn");
  const backBtn = $("back-btn");

  // ---------- Helpers ----------
  function showError(el, msg) {
    if (!msg) {
      el.style.display = "none";
      el.textContent = "";
    } else {
      el.style.display = "block";
      el.textContent = msg;
    }
  }

  function go(page) {
    if (pageDashboard)    pageDashboard.style.display    = page === "dashboard"    ? "" : "none";
    if (pageUpload)       pageUpload.style.display       = page === "upload"       ? "" : "none";
    if (pageConfig)       pageConfig.style.display       = page === "config"       ? "" : "none";
    if (pageRecord)       pageRecord.style.display       = page === "record"       ? "" : "none";
    if (pageArchitecture) pageArchitecture.style.display = page === "architecture" ? "" : "none";
    if (pageFila)         pageFila.style.display         = page === "fila"         ? "" : "none";
    navTabs.forEach((t) => {
      t.classList.toggle("active", t.dataset.page === page);
    });
    // Per-page hooks
    if (page === "dashboard") loadDashboard();
    if (page === "upload")    refreshProcessingStatus();
    if (page === "fila")      loadFila();
    // Scroll back to the top when switching pages
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const page = tab.dataset.page;
      // Navigating to "record" through the tab does not trigger the warmup or reset state.
      // The "Continue" button on screen 1 does that.
      go(page);
    });
  });

  function formatTimer(sec) {
    const m = String(Math.floor(sec / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  function updateContinueState() {
    const hasCustom = state.customQuestions.some((q) => q.trim());
    continueBtn.disabled = state.selectedPresets.size === 0 && !hasCustom;
  }

  // ---------- Screen 1: presets ----------
  async function loadPresets() {
    try {
      const r = await fetch("/api/presets");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      state.presets = await r.json();
      renderPresets();
      updateContinueState();
    } catch (e) {
      showError(configError, `Error loading presets: ${e.message}`);
    }
  }

  function renderPresets() {
    presetsGrid.innerHTML = "";
    for (const p of state.presets) {
      const label = document.createElement("label");
      label.className =
        "preset" + (state.selectedPresets.has(p.key) ? " checked" : "");
      label.innerHTML = `
        <input type="checkbox" ${state.selectedPresets.has(p.key) ? "checked" : ""} />
        <div class="preset-text">
          <div class="preset-label"></div>
          <div class="preset-desc"></div>
        </div>
      `;
      // Text via textContent to avoid XSS
      label.querySelector(".preset-label").textContent = p.label;
      label.querySelector(".preset-desc").textContent = p.description || "";
      label.querySelector("input").addEventListener("change", () => {
        if (state.selectedPresets.has(p.key)) {
          state.selectedPresets.delete(p.key);
          label.classList.remove("checked");
        } else {
          state.selectedPresets.add(p.key);
          label.classList.add("checked");
        }
        updateContinueState();
      });
      presetsGrid.appendChild(label);
    }
  }

  function renderQuestions() {
    questionsList.innerHTML = "";
    state.customQuestions.forEach((q, i) => {
      const row = document.createElement("div");
      row.className = "custom-row";
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = `Question ${i + 1}`;
      input.value = q;
      input.addEventListener("input", (ev) => {
        state.customQuestions[i] = ev.target.value;
        updateContinueState();
      });
      row.appendChild(input);
      if (state.customQuestions.length > 1) {
        const rmBtn = document.createElement("button");
        rmBtn.type = "button";
        rmBtn.className = "ghost";
        rmBtn.textContent = "×";
        rmBtn.setAttribute("aria-label", "Remove");
        rmBtn.addEventListener("click", () => {
          state.customQuestions.splice(i, 1);
          renderQuestions();
          updateContinueState();
        });
        row.appendChild(rmBtn);
      }
      questionsList.appendChild(row);
    });
  }

  addQuestionBtn.addEventListener("click", () => {
    state.customQuestions.push("");
    renderQuestions();
  });

  continueBtn.addEventListener("click", () => {
    showError(recError, null);
    state.transcript = "";
    transcriptBox.textContent = "The transcript will appear here as you speak…";
    transcriptBox.classList.add("empty");
    analysisEmpty.style.display = "";
    analysisResult.style.display = "none";
    analysisLoading.style.display = "none";
    showError(analysisError, null);
    analyzeBtn.disabled = true;
    timer.textContent = "00:00";
    recStatus.textContent = "Warming up the model…";
    recStatus.classList.remove("recording");
    go("record");

    // Fire the warmup in parallel. If the first access takes ~30s, this keeps
    // the transcript from losing the first seconds of speech to a cold start.
    fetch("/api/warmup", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "warm") {
          recStatus.textContent = "Ready to record";
        } else {
          recStatus.textContent = "Ready to record (model may be cold)";
        }
      })
      .catch(() => {
        recStatus.textContent = "Ready to record";
      });
  });

  backBtn.addEventListener("click", () => {
    if (state.recording) stopRecording().catch(() => {});
    go("config");
  });

  // ---------- Screen 2: recording + WebSocket ----------
  function buildWsUrl(path) {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}${path}`;
  }

  async function startRecording() {
    showError(recError, null);
    state.transcript = "";
    transcriptBox.textContent = "The transcript will appear here as you speak…";
    transcriptBox.classList.add("empty");
    analyzeBtn.disabled = true;
    analysisEmpty.style.display = "";
    analysisResult.style.display = "none";

    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
          sampleRate: 16000,
        },
      });

      const ws = new WebSocket(buildWsUrl("/ws/transcribe"));
      ws.binaryType = "arraybuffer";
      state.ws = ws;
      await new Promise((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("WebSocket failed to connect"));
      });

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === "partial" || data.type === "final") {
            state.transcript = data.text || "";
            if (state.transcript) {
              transcriptBox.textContent = state.transcript;
              transcriptBox.classList.remove("empty");
            }
            // Subtly flash the status to signal an update
            if (state.recording) {
              recStatus.textContent = "● Recording — updating…";
              setTimeout(() => {
                if (state.recording) {
                  recStatus.textContent = "● Recording — click to stop";
                }
              }, 600);
            }
          } else if (data.type === "error") {
            showError(recError, data.message || "Transcription error");
          }
        } catch {}
      };

      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((m) =>
        MediaRecorder.isTypeSupported(m)
      );
      if (!mime) throw new Error("Browser does not support MediaRecorder with WebM/Opus");

      const recorder = new MediaRecorder(state.stream, {
        mimeType: mime,
        audioBitsPerSecond: 64_000,
      });
      state.recorder = recorder;
      recorder.ondataavailable = async (ev) => {
        if (ev.data && ev.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          const buf = await ev.data.arrayBuffer();
          ws.send(buf);
        }
      };
      recorder.start(1000);

      state.startTs = Date.now();
      state.timerInterval = setInterval(() => {
        timer.textContent = formatTimer(
          Math.floor((Date.now() - state.startTs) / 1000)
        );
      }, 250);

      state.recording = true;
      recButton.classList.add("recording");
      recButton.setAttribute("aria-label", "Stop recording");
      recStatus.textContent = "● Recording — click to stop";
      recStatus.classList.add("recording");
    } catch (e) {
      showError(recError, e.message || "Failed to start recording");
      await stopRecording().catch(() => {});
    }
  }

  async function stopRecording() {
    if (!state.recording && !state.recorder && !state.ws && !state.stream) return;
    state.recording = false;
    recButton.classList.remove("recording");
    recButton.setAttribute("aria-label", "Start recording");
    recStatus.textContent = "Finishing…";
    recStatus.classList.remove("recording");

    if (state.recorder && state.recorder.state !== "inactive") {
      await new Promise((resolve) => {
        const r = state.recorder;
        r.onstop = () => resolve();
        try {
          r.stop();
        } catch {
          resolve();
        }
      });
    }
    state.recorder = null;

    await new Promise((r) => setTimeout(r, 250));

    const ws = state.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "end" }));
      // Wait for the 'final' message (15s timeout)
      await new Promise((resolve) => {
        const t = setTimeout(resolve, 15_000);
        const handler = (ev) => {
          try {
            const data = JSON.parse(ev.data);
            if (data.type === "final") {
              clearTimeout(t);
              ws.removeEventListener("message", handler);
              resolve();
            }
          } catch {}
        };
        ws.addEventListener("message", handler);
      });
    }

    if (state.timerInterval) {
      clearInterval(state.timerInterval);
      state.timerInterval = null;
    }
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    if (ws) {
      try {
        ws.close();
      } catch {}
    }
    state.ws = null;

    recStatus.textContent = "Recording complete";
    if (state.transcript.trim()) {
      analyzeBtn.disabled = false;
      analysisEmpty.textContent =
        "Click 'Analyze transcript' to extract the configured insights.";
    }
  }

  recButton.addEventListener("click", async () => {
    if (state.recording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  });

  // ---------- LLM analysis ----------
  const FIELD_LABELS = {
    sentiment: "Sentiment",
    emotion: "Emotion",
    intent: "Intent",
    topics: "Topics",
    churn_risk: "Churn risk",
    compliance_ok: "Compliance",
    summary: "Summary",
    next_action: "Next action",
  };

  function renderValue(value) {
    if (value === null || value === undefined) {
      const e = document.createElement("em");
      e.textContent = "—";
      return e;
    }
    if (Array.isArray(value)) {
      const frag = document.createDocumentFragment();
      for (const item of value) {
        const span = document.createElement("span");
        span.className = "pill";
        span.textContent = String(item);
        frag.appendChild(span);
      }
      return frag;
    }
    if (typeof value === "boolean") {
      const span = document.createElement("span");
      span.className = "pill " + (value ? "green" : "red");
      span.textContent = value ? "Yes" : "No";
      return span;
    }
    if (typeof value === "number") {
      return document.createTextNode(value.toFixed(2));
    }
    if (typeof value === "string") {
      return document.createTextNode(value);
    }
    const pre = document.createElement("pre");
    pre.style.fontSize = "12px";
    pre.textContent = JSON.stringify(value, null, 2);
    return pre;
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!state.transcript.trim()) return;
    showError(analysisError, null);
    analysisEmpty.style.display = "none";
    analysisResult.style.display = "none";
    analysisLoading.style.display = "";
    analyzeBtn.disabled = true;

    try {
      const cleanQuestions = state.customQuestions
        .map((q) => q.trim())
        .filter(Boolean);
      const r = await fetch("/api/analyze", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          transcript: state.transcript,
          presets: Array.from(state.selectedPresets),
          custom_questions: cleanQuestions,
        }),
      });
      if (!r.ok) {
        const text = await r.text();
        throw new Error(`HTTP ${r.status}: ${text.slice(0, 200)}`);
      }
      const data = await r.json();
      const result = data.result || {};

      analysisResult.innerHTML = "";
      const cleanQuestionsForLabel = state.customQuestions
        .map((q) => q.trim())
        .filter(Boolean);
      for (const [key, value] of Object.entries(result)) {
        const item = document.createElement("div");
        const wide =
          key === "summary" ||
          key === "next_action" ||
          key === "topics" ||
          key.startsWith("q");
        item.className = "analysis-item" + (wide ? " full" : "");

        const k = document.createElement("div");
        k.className = "analysis-key";
        if (key.startsWith("q")) {
          const idx = parseInt(key.slice(1), 10) - 1;
          k.textContent =
            cleanQuestionsForLabel[idx] || FIELD_LABELS[key] || key;
        } else {
          k.textContent = FIELD_LABELS[key] || key;
        }
        item.appendChild(k);

        const v = document.createElement("div");
        v.className = "analysis-value";
        v.appendChild(renderValue(value));
        item.appendChild(v);
        analysisResult.appendChild(item);
      }
      analysisResult.style.display = "";
    } catch (e) {
      showError(analysisError, e.message || "Analysis error");
    } finally {
      analysisLoading.style.display = "none";
      analyzeBtn.disabled = false;
    }
  });

  // ============================================================
  // Aurélia portal — Dashboard + Upload + Drill-down
  // ============================================================

  const dashRefs = {
    refresh: $("dash-refresh"),
    updated: $("dash-updated"),
    kpiTotal:    $("kpi-total"),
    kpiTotalSub: $("kpi-total-sub"),
    kpiAnalyzed:    $("kpi-analyzed"),
    kpiAnalyzedSub: $("kpi-analyzed-sub"),
    kpiPending:  $("kpi-pending"),
    kpiFraud:    $("kpi-fraud"),
    kpiFraudSub: $("kpi-fraud-sub"),
    kpiRein:     $("kpi-rein"),
    kpiChurn:    $("kpi-churn"),
    chartTipo: $("chart-tipo"),
    chartSent: $("chart-sent"),
    topCons:   $("top-cons"),
    topProd:   $("top-prod"),
    callsList: $("calls-list"),
    filterTipo: $("filter-tipo"),
    filterFraude: $("filter-fraude"),
    filterSource: $("filter-source"),
  };

  const TIPO_LABEL = {
    atraso: "Late delivery",
    falta_parcial: "Missing items",
    violacao_caixa: "Damaged or opened box",
    fraude_suspeita: "Suspected fraud",
    outro: "Other",
    sem_classificacao: "Unclassified",
  };
  const SENT_LABEL = {
    positivo: "positive",
    neutro: "neutral",
    negativo: "negative",
  };

  function fmtPct(n, total) {
    if (!total) return "—";
    return ((100 * n) / total).toFixed(0) + "%";
  }

  function renderBarChart(container, rows, keyName) {
    container.innerHTML = "";
    if (!rows || !rows.length) {
      container.innerHTML = '<p style="color: var(--db-muted)">No data yet</p>';
      return;
    }
    const max = Math.max(...rows.map((r) => r.n));
    rows.forEach((r) => {
      const row = document.createElement("div");
      row.className = "chart-row";
      const labelKey = keyName === "tipo" ? r.tipo : r.sentiment;
      const niceLabel =
        keyName === "tipo" ? (TIPO_LABEL[labelKey] || labelKey) : (SENT_LABEL[labelKey] || labelKey);

      const lab = document.createElement("div");
      lab.textContent = niceLabel;
      const bar = document.createElement("div");
      bar.className = "chart-bar-bg";
      const fill = document.createElement("div");
      fill.className = "chart-bar-fill " + labelKey;
      fill.style.width = (max > 0 ? (100 * r.n) / max : 0) + "%";
      bar.appendChild(fill);
      const cnt = document.createElement("div");
      cnt.className = "chart-bar-count";
      cnt.textContent = r.n;

      row.appendChild(lab);
      row.appendChild(bar);
      row.appendChild(cnt);
      container.appendChild(row);
    });
  }

  function renderTable(container, headers, rows) {
    container.innerHTML = "";
    if (!rows || !rows.length) {
      container.innerHTML = '<p style="color: var(--db-muted)">No data yet</p>';
      return;
    }
    const t = document.createElement("table");
    t.className = "data-table";
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    headers.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h.label;
      if (h.num) th.classList.add("num");
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    t.appendChild(thead);
    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      headers.forEach((h) => {
        const td = document.createElement("td");
        if (h.num) td.classList.add("num");
        td.textContent = row[h.key] ?? "—";
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    t.appendChild(tbody);
    container.appendChild(t);
  }

  function renderCallsList(calls) {
    dashRefs.callsList.innerHTML = "";
    if (!calls.length) {
      dashRefs.callsList.innerHTML =
        '<p style="color: var(--db-muted)">No calls match these filters.</p>';
      return;
    }
    calls.forEach((c) => {
      const row = document.createElement("div");
      row.className = "call-row";
      row.dataset.callId = c.call_id;

      const id = document.createElement("div");
      id.className = "call-id";
      id.title = c.call_id;
      id.textContent = c.call_id;

      const prev = document.createElement("div");
      prev.className = "call-preview";
      prev.textContent =
        c.transcript_preview ||
        (c.analyzed_at ? "(no transcript)" : "Awaiting processing…");

      // Badges
      const badges = document.createElement("div");
      badges.style.display = "flex";
      badges.style.gap = "6px";
      if (c.tipo_problema) {
        const b = document.createElement("span");
        b.className = "call-badge " + c.tipo_problema;
        b.textContent = TIPO_LABEL[c.tipo_problema] || c.tipo_problema;
        badges.appendChild(b);
      } else if (!c.analyzed_at) {
        const b = document.createElement("span");
        b.className = "call-badge pendente";
        b.textContent = "pending";
        badges.appendChild(b);
      }
      if (c.suspeita_fraude) {
        const b = document.createElement("span");
        b.className = "call-badge fraude";
        b.textContent = "fraud";
        badges.appendChild(b);
      }
      if (c.reincidencia) {
        const b = document.createElement("span");
        b.className = "call-badge rein";
        b.textContent = "repeat issue";
        badges.appendChild(b);
      }

      const churn = document.createElement("div");
      churn.style.fontSize = "12px";
      churn.style.color = "var(--db-muted)";
      churn.textContent =
        c.churn_risk != null ? "churn " + Number(c.churn_risk).toFixed(2) : "";

      const date = document.createElement("div");
      date.style.fontSize = "11px";
      date.style.color = "var(--db-muted)";
      const ts = c.analyzed_at || c.ingested_at || "";
      date.textContent = ts ? ts.slice(0, 16).replace("T", " ") : "—";

      row.appendChild(id);
      row.appendChild(prev);
      row.appendChild(badges);
      row.appendChild(churn);
      row.appendChild(date);
      row.addEventListener("click", () => openCallModal(c.call_id));
      dashRefs.callsList.appendChild(row);
    });
  }

  async function loadDashboard() {
    if (dashRefs.updated) dashRefs.updated.textContent = "refreshing…";
    try {
      const r = await fetch("/api/portal/dashboard");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      const k = d.kpis;

      dashRefs.kpiTotal.textContent    = k.total_calls;
      dashRefs.kpiAnalyzed.textContent = k.analyzed_calls;
      dashRefs.kpiAnalyzedSub.textContent = fmtPct(k.analyzed_calls, k.total_calls) + " of total";
      dashRefs.kpiPending.textContent  = k.pending_calls;
      dashRefs.kpiFraud.textContent    = k.suspeita_fraude_count;
      dashRefs.kpiFraudSub.textContent = fmtPct(k.suspeita_fraude_count, k.analyzed_calls) + " of analyzed";
      dashRefs.kpiRein.textContent     = k.reincidencia_count;
      dashRefs.kpiChurn.textContent    = (k.avg_churn_risk || 0).toFixed(2);

      renderBarChart(dashRefs.chartTipo, d.by_tipo_problema, "tipo");
      renderBarChart(dashRefs.chartSent, d.by_sentiment, "sentiment");

      renderTable(
        dashRefs.topCons,
        [
          { key: "consultora_codigo", label: "Consultant" },
          { key: "chamadas",          label: "Calls",          num: true },
          { key: "fraudes",           label: "Fraud",          num: true },
          { key: "reincidencias",     label: "Repeat",         num: true },
        ],
        d.top_consultoras,
      );
      renderTable(
        dashRefs.topProd,
        [
          { key: "produto", label: "Product" },
          { key: "n",       label: "Calls", num: true },
        ],
        d.top_produtos_faltantes,
      );

      // Default list
      await loadCallsList();

      const now = new Date();
      dashRefs.updated.textContent = "updated " + now.toLocaleTimeString("en-US");
    } catch (e) {
      console.error(e);
      dashRefs.updated.textContent = "error: " + e.message;
    }
  }

  async function loadCallsList() {
    const params = new URLSearchParams();
    if (dashRefs.filterTipo.value)   params.set("tipo_problema", dashRefs.filterTipo.value);
    if (dashRefs.filterFraude.value) params.set("suspeita_fraude", dashRefs.filterFraude.value);
    if (dashRefs.filterSource.value) params.set("source", dashRefs.filterSource.value);
    params.set("limit", "100");
    try {
      const r = await fetch("/api/portal/calls?" + params.toString());
      if (!r.ok) throw new Error("HTTP " + r.status);
      const calls = await r.json();
      renderCallsList(calls);
    } catch (e) {
      dashRefs.callsList.innerHTML =
        '<p style="color: var(--db-red)">Failed to list calls: ' + e.message + "</p>";
    }
  }

  if (dashRefs.refresh) dashRefs.refresh.addEventListener("click", loadDashboard);
  if (dashRefs.filterTipo)   dashRefs.filterTipo.addEventListener("change", loadCallsList);
  if (dashRefs.filterFraude) dashRefs.filterFraude.addEventListener("change", loadCallsList);
  if (dashRefs.filterSource) dashRefs.filterSource.addEventListener("change", loadCallsList);

  // Clickable KPIs: clicking filters the list below and scrolls to it.
  document.querySelectorAll(".kpi-card.clickable").forEach((card) => {
    card.addEventListener("click", () => {
      const f = card.dataset.filter;
      if (f === "fraude" && dashRefs.filterFraude) {
        dashRefs.filterFraude.value = "true";
        if (dashRefs.filterTipo)   dashRefs.filterTipo.value = "";
        if (dashRefs.filterSource) dashRefs.filterSource.value = "";
        loadCallsList();
        dashRefs.callsList.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });

  // ---------- Drill-down modal ----------
  const cmRefs = {
    modal: $("call-modal"),
    close: $("cm-close"),
    title: $("cm-title"),
    audio: $("cm-audio"),
    format: $("cm-format"),
    source: $("cm-source"),
    ingested: $("cm-ingested"),
    transcript: $("cm-transcript"),
    tipo: $("cm-tipo"),
    fraude: $("cm-fraude"),
    fraudeConf: $("cm-fraude-conf"),
    fraudeConfBar: $("cm-fraude-conf-bar"),
    rein: $("cm-rein"),
    churn: $("cm-churn"),
    cons: $("cm-cons"),
    ped: $("cm-ped"),
    sent: $("cm-sent"),
    emo: $("cm-emo"),
    int: $("cm-int"),
    comp: $("cm-comp"),
    compReason: $("cm-comp-reason"),
    fraudeInd: $("cm-fraude-ind"),
    prod: $("cm-prod"),
    topics: $("cm-topics"),
    summary: $("cm-summary"),
    next: $("cm-next"),
  };

  if (cmRefs.close) cmRefs.close.addEventListener("click", closeCallModal);
  if (cmRefs.modal) {
    cmRefs.modal.addEventListener("click", (e) => {
      if (e.target === cmRefs.modal) closeCallModal();
    });
  }

  // The color band exists because 18% and 92% look too similar as plain text.
  function setFraudeConf(valor) {
    const bar = cmRefs.fraudeConfBar;
    const num = Number(valor);
    if (valor == null || Number.isNaN(num)) {
      cmRefs.fraudeConf.textContent = "—";
      bar.style.width = "0%";
      bar.className = "conf-bar";
      return;
    }
    const pct = Math.round(Math.max(0, Math.min(1, num)) * 100);
    cmRefs.fraudeConf.textContent = pct + "%";
    bar.style.width = pct + "%";
    bar.className = "conf-bar" + (pct >= 70 ? " alta" : pct >= 40 ? " media" : "");
  }

  function fillBullets(ul, items) {
    ul.innerHTML = "";
    if (!items || !items.length) {
      const li = document.createElement("li");
      li.style.color = "var(--db-muted)";
      li.textContent = "—";
      ul.appendChild(li);
      return;
    }
    items.forEach((it) => {
      const li = document.createElement("li");
      li.textContent = it;
      ul.appendChild(li);
    });
  }

  async function openCallModal(callId) {
    cmRefs.title.textContent = callId;
    cmRefs.transcript.textContent = "Loading…";
    setFraudeConf(null);
    cmRefs.audio.src = "/api/portal/audio/" + encodeURIComponent(callId);
    cmRefs.modal.style.display = "";
    document.body.style.overflow = "hidden";
    try {
      const r = await fetch("/api/portal/calls/" + encodeURIComponent(callId));
      if (!r.ok) throw new Error("HTTP " + r.status);
      const d = await r.json();
      cmRefs.format.textContent = (d.audio_format || "").toUpperCase();
      cmRefs.source.textContent = d.source || "—";
      cmRefs.ingested.textContent = d.ingested_at ? d.ingested_at.slice(0, 19).replace("T", " ") : "—";
      cmRefs.transcript.textContent = d.full_text || "(transcript not available — it may still be processing)";
      cmRefs.tipo.textContent  = TIPO_LABEL[d.tipo_problema] || d.tipo_problema || "—";
      cmRefs.fraude.textContent = d.suspeita_fraude == null ? "—" : (d.suspeita_fraude ? "Yes" : "No");
      setFraudeConf(d.fraude_confianca);
      cmRefs.rein.textContent   = d.reincidencia == null   ? "—" : (d.reincidencia   ? "Yes" : "No");
      cmRefs.churn.textContent  = d.churn_risk != null ? Number(d.churn_risk).toFixed(2) : "—";
      cmRefs.cons.textContent   = d.consultora_codigo || "—";
      cmRefs.ped.textContent    = d.pedido_numero || "—";
      cmRefs.sent.textContent   = SENT_LABEL[d.sentiment] || d.sentiment || "—";
      cmRefs.emo.textContent    = d.emotion   || "—";
      cmRefs.int.textContent    = d.intent    || "—";
      cmRefs.comp.textContent   = d.compliance_ok == null ? "—" : (d.compliance_ok ? "OK" : "Failed");
      cmRefs.compReason.textContent = d.compliance_reason || "";
      fillBullets(cmRefs.fraudeInd, d.fraude_indicadores);
      fillBullets(cmRefs.prod, d.produtos_faltantes);
      fillBullets(cmRefs.topics, d.topics);
      cmRefs.summary.textContent = d.summary || "—";
      cmRefs.next.textContent = d.next_action || "—";
    } catch (e) {
      cmRefs.transcript.textContent = "Error loading: " + e.message;
    }
  }

  function closeCallModal() {
    cmRefs.modal.style.display = "none";
    cmRefs.audio.pause();
    cmRefs.audio.src = "";
    document.body.style.overflow = "";
  }

  // ---------- Upload ----------
  const upRefs = {
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    list: $("upload-list"),
    btnProc: $("btn-process"),
    procLog: $("proc-log"),
    procPill: $("proc-status-pill"),
  };

  function logProc(msg, klass) {
    if (!upRefs.procLog) return;
    if (upRefs.procLog.querySelector(".muted")) upRefs.procLog.innerHTML = "";
    const line = document.createElement("div");
    if (klass) line.className = klass;
    const ts = new Date().toLocaleTimeString("en-US");
    line.textContent = "[" + ts + "] " + msg;
    upRefs.procLog.appendChild(line);
    upRefs.procLog.scrollTop = upRefs.procLog.scrollHeight;
  }

  function fmtSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / 1024 / 1024).toFixed(1) + " MB";
  }

  async function uploadFile(file) {
    const row = document.createElement("div");
    row.className = "upload-row";
    const name = document.createElement("div");
    name.textContent = file.name;
    const size = document.createElement("div");
    size.textContent = fmtSize(file.size);
    const status = document.createElement("div");
    status.className = "upload-status busy";
    status.textContent = "uploading…";
    row.appendChild(name);
    row.appendChild(size);
    row.appendChild(status);
    upRefs.list.appendChild(row);

    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const r = await fetch("/api/portal/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error("HTTP " + r.status + ": " + (await r.text()).slice(0, 200));
      status.className = "upload-status ok";
      status.textContent = "✓ in the volume";
    } catch (e) {
      status.className = "upload-status error";
      status.textContent = "error: " + e.message;
    }
  }

  async function uploadFiles(files) {
    for (const f of files) {
      // sequential so we don't saturate the connection (files can be large)
      await uploadFile(f);
    }
  }

  if (upRefs.dropzone) {
    upRefs.dropzone.addEventListener("click", () => upRefs.fileInput.click());
    upRefs.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      upRefs.dropzone.classList.add("drag-over");
    });
    upRefs.dropzone.addEventListener("dragleave", () => {
      upRefs.dropzone.classList.remove("drag-over");
    });
    upRefs.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      upRefs.dropzone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        uploadFiles(Array.from(e.dataTransfer.files));
      }
    });
    upRefs.fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length) {
        uploadFiles(Array.from(e.target.files));
      }
    });
  }

  // Processing
  let procPolling = null;

  async function refreshProcessingStatus() {
    try {
      const r = await fetch("/api/portal/processing");
      if (!r.ok) return;
      const s = await r.json();
      if (upRefs.procPill) upRefs.procPill.textContent = s.status;
      if (upRefs.procPill) {
        upRefs.procPill.className = "pill " + (
          s.status === "running" ? "blue" :
          s.status === "success" ? "green" :
          s.status === "failed"  ? "red"   : ""
        );
      }
      if (s.status === "running" && !procPolling) {
        procPolling = setInterval(refreshProcessingStatus, 5000);
        logProc("Pipeline running (run_id=" + (s.run_id || "?") + ")", "info");
      }
      if (s.status !== "running" && procPolling) {
        clearInterval(procPolling);
        procPolling = null;
        const klass = s.status === "success" ? "ok" : "err";
        logProc("Pipeline " + s.status + " — " + (s.message || ""), klass);
        // reload the dashboard if it is visible
        if (pageDashboard.style.display !== "none") loadDashboard();
      }
    } catch (e) {
      console.error(e);
    }
  }

  if (upRefs.btnProc) {
    upRefs.btnProc.addEventListener("click", async () => {
      upRefs.btnProc.disabled = true;
      try {
        const r = await fetch("/api/portal/process", { method: "POST" });
        const d = await r.json();
        logProc("→ " + (d.message || d.status), d.status === "started" ? "info" : "err");
        await refreshProcessingStatus();
      } catch (e) {
        logProc("Error: " + e.message, "err");
      } finally {
        upRefs.btnProc.disabled = false;
      }
    });
  }

  // ---------- Investigation queue (Lakebase) ----------
  const DECISOES = {
    fraude_confirmada: "Confirmed fraud",
    procedente: "Valid claim",
    improcedente: "Unfounded",
    pedir_evidencia: "Request evidence",
  };
  const SINAIS = {
    detalhes_vagos_ou_contraditorios: "vague or contradictory details",
    pressa_pelo_ressarcimento: "rushing the reimbursement",
    resiste_a_enviar_evidencia: "resists sending evidence",
    alega_nao_ter_conferido_na_entrega: "says she did not check at delivery",
    historico_de_azar_recorrente: "recurring bad luck",
    acusa_terceiros_sem_evidencia: "blames others without evidence",
  };
  const FAIXA_LABEL = {
    alta: "high",
    media: "medium",
    baixa: "low",
  };
  const MOTIVO_LABEL = {
    falta_parcial: "missing items",
    nao_recebido: "not received",
    caixa_violada: "opened box",
    atraso: "late delivery",
    produto_avariado: "damaged product",
    outro: "other",
  };
  const motivo = (m) => MOTIVO_LABEL[m] || m;
  let filaSelecionado = null;

  function el(tag, cls, texto) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (texto !== undefined && texto !== null) e.textContent = texto;
    return e;
  }
  const brl = (v) => Number(v || 0).toLocaleString("en-US", { style: "currency", currency: "BRL" });
  const corScore = (s) => (s >= 0.5 ? "var(--db-red)" : s >= 0.2 ? "var(--db-orange)" : "var(--db-green)");

  async function loadFila() {
    try {
      const [rRes, rLista] = await Promise.all([
        fetch("/api/fila/resumo"),
        fetch("/api/fila?" + new URLSearchParams({
          ...($("fila-faixa").value ? { faixa: $("fila-faixa").value } : {}),
          pendentes: $("fila-pendentes").checked,
          limite: 200,
        })),
      ]);
      if (!rRes.ok || !rLista.ok) throw new Error("HTTP " + (rRes.ok ? rLista.status : rRes.status));
      const resumo = await rRes.json();
      $("fk-abertos").textContent = resumo.abertos;
      $("fk-alto").textContent = resumo.alto_risco;
      $("fk-alto-sub").textContent = brl(resumo.valor_alto_risco) + " claimed";
      $("fk-decididos").textContent = resumo.decididos;
      $("fk-fraudes").textContent = resumo.fraudes_confirmadas;
      $("fila-modelo").textContent = "model v" + resumo.modelo_versao;
      renderFila(await rLista.json());
    } catch (e) {
      $("fila-lista").innerHTML = "";
      $("fila-lista").appendChild(el("p", "muted", "Could not read the queue: " + e.message));
    }
  }

  function renderFila(linhas) {
    const box = $("fila-lista");
    box.innerHTML = "";
    if (!linhas.length) { box.appendChild(el("p", "muted", "Nothing in the queue with this filter.")); return; }
    const t = el("table", "data-table");
    const h = el("tr");
    ["#", "Score", "Claim", "Motive", "Amount", "Route", "Call"].forEach((x) => h.appendChild(el("th", null, x)));
    t.appendChild(el("thead")).appendChild(h);
    const tb = el("tbody");
    linhas.forEach((l) => {
      const tr = el("tr", "fila-row" + (l.chamado_id === filaSelecionado ? " selecionada" : ""));
      tr.appendChild(el("td", "num", l.posicao));
      const tdScore = el("td");
      const sc = el("div", "score-cell");
      const bar = el("div", "score-bar");
      const fill = el("span");
      fill.style.width = Math.round(l.score_fraude * 100) + "%";
      fill.style.background = corScore(l.score_fraude);
      bar.appendChild(fill);
      sc.appendChild(bar);
      sc.appendChild(el("span", null, Number(l.score_fraude).toFixed(2)));
      tdScore.appendChild(sc);
      tr.appendChild(tdScore);
      const tdCh = el("td", null, l.chamado_id);
      if (l.decisao) tdCh.appendChild(el("span", "pill " + (l.decisao === "fraude_confirmada" ? "red" : "green"), " " + DECISOES[l.decisao]));
      tr.appendChild(tdCh);
      tr.appendChild(el("td", null, motivo(l.motivo_declarado)));
      tr.appendChild(el("td", "num", brl(l.valor_reclamado)));
      tr.appendChild(el("td", null, l.transportadora + " · " + l.uf));
      tr.appendChild(el("td", null, l.call_id ? (l.sinais_relato.length ? "⚠ " + l.sinais_relato.length + " signal(s)" : "no signals") : "—"));
      tr.addEventListener("click", () => abrirChamado(l.chamado_id));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    box.appendChild(t);
  }

  async function abrirChamado(id) {
    filaSelecionado = id;
    document.querySelectorAll(".fila-row").forEach((r) => r.classList.toggle("selecionada", r.firstChild && r.children[2].firstChild.textContent === id));
    const box = $("fila-detalhe");
    box.innerHTML = "";
    box.appendChild(el("p", "muted", "Loading " + id + "…"));
    let c;
    try {
      const r = await fetch("/api/fila/" + encodeURIComponent(id));
      if (!r.ok) throw new Error("HTTP " + r.status);
      c = await r.json();
    } catch (e) {
      box.innerHTML = "";
      box.appendChild(el("p", "muted", "Failed to open: " + e.message));
      return;
    }
    box.innerHTML = "";
    const topo = el("div", "score-cell");
    topo.appendChild(el("h3", null, c.chamado_id));
    const pill = el("span", "pill " + (c.faixa_risco === "alta" ? "red" : c.faixa_risco === "media" ? "orange" : "green"),
      (FAIXA_LABEL[c.faixa_risco] || c.faixa_risco) + " risk · " + Number(c.score_fraude).toFixed(2));
    topo.appendChild(pill);
    box.appendChild(topo);
    box.appendChild(el("span", "muted", "Calibrated fraud probability, model v" + c.modelo_versao + ". Position " + c.posicao + " in the queue."));

    box.appendChild(el("h4", null, "The claim"));
    const dl = el("dl", "fila-fatos");
    const div = (v, ref) => {
      if (v === null || v === undefined) return "no weighing";
      const pct = Number(v) * 100;
      if (Math.abs(pct) < 0.05) return "same as " + ref;
      return Math.abs(pct).toFixed(1) + "% " + (pct > 0 ? "below " : "above ") + ref;
    };
    [
      ["Consultant", c.consultora_codigo + " · " + Number(c.meses_casa).toFixed(0) + " months with the company"],
      ["Order", c.pedido_numero],
      ["Motive", motivo(c.motivo_declarado) + " · " + brl(c.valor_reclamado)],
      ["Delivery", c.transportadora + " · " + c.uf + " · " + c.cd + (c.comprovante_assinado ? " · signed proof of delivery" : "")],
      ["Weight at DC", div(c.div_peso_cd, "expected")],
      ["Weight at hub", div(c.div_peso_hub, "the shipped weight")],
      ["History", c.chamados_180d + " claim(s) in 180 days · " + c.fraudes_confirmadas_antes + " confirmed fraud(s) before"],
    ].forEach(([k, v]) => { dl.appendChild(el("dt", null, k)); dl.appendChild(el("dd", null, v)); });
    box.appendChild(dl);

    if (c.call_id) {
      box.appendChild(el("h4", null, "The call"));
      if (c.sinais_relato.length) {
        const s = el("div", "fila-sinais");
        c.sinais_relato.forEach((x) => s.appendChild(el("span", "pill orange", SINAIS[x] || x)));
        box.appendChild(s);
      } else {
        box.appendChild(el("span", "muted", "No signals in the consultant's account."));
      }
      if (c.resumo_ligacao) box.appendChild(el("p", null, c.resumo_ligacao));
      if (c.tem_audio) {
        const a = el("audio");
        a.controls = true;
        a.preload = "none";
        a.src = "/api/portal/audio/" + encodeURIComponent(c.call_id);
        a.style.width = "100%";
        box.appendChild(a);
      }
      if (c.transcricao) {
        const det = el("details");
        det.appendChild(el("summary", "muted", "Full transcript (Portuguese)"));
        det.appendChild(el("div", "fila-transcricao", c.transcricao));
        box.appendChild(det);
      }
    }

    box.appendChild(el("h4", null, "Your decision"));
    const coment = el("textarea");
    coment.placeholder = "Comment (optional): what you checked, what is still missing";
    const btns = el("div", "decisao-btns");
    [["fraude_confirmada", "fraude"], ["procedente", "procedente"], ["improcedente", "ghost"], ["pedir_evidencia", "ghost"]].forEach(([k, cls]) => {
      const b = el("button", cls, DECISOES[k]);
      b.addEventListener("click", () => decidir(c.chamado_id, k, coment.value, btns));
      btns.appendChild(b);
    });
    box.appendChild(btns);
    box.appendChild(coment);
    if (c.historico && c.historico.length) {
      const hist = el("div", "fila-historico");
      c.historico.forEach((d) => hist.appendChild(el("div", null,
        new Date(d.decidido_em).toLocaleString("en-US") + " · " + DECISOES[d.decisao] + " · " + d.analista + (d.comentario ? " — " + d.comentario : ""))));
      box.appendChild(hist);
    }
  }

  async function decidir(id, decisao, comentario, btns) {
    btns.querySelectorAll("button").forEach((b) => (b.disabled = true));
    try {
      const r = await fetch("/api/fila/" + encodeURIComponent(id) + "/decisao", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decisao, comentario: comentario || null }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || "HTTP " + r.status);
      await loadFila();
      await abrirChamado(id);
    } catch (e) {
      alert("Could not save the decision: " + e.message);
      btns.querySelectorAll("button").forEach((b) => (b.disabled = false));
    }
  }

  $("fila-refresh").addEventListener("click", loadFila);
  $("fila-faixa").addEventListener("change", loadFila);
  $("fila-pendentes").addEventListener("change", loadFila);

  // Without a Whisper endpoint, live recording does not work: hide its tabs.
  fetch("/api/recursos").then((r) => r.json()).then((rec) => {
    if (!rec.gravacao_ao_vivo) {
      document.querySelectorAll('.nav-tab[data-page="config"], .nav-tab[data-page="record"]').forEach((b) => (b.style.display = "none"));
    }
  }).catch(() => {});

  // ---------- Init ----------
  renderQuestions();
  loadPresets();
  go("dashboard");
})();
