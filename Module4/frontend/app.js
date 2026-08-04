// Module 4 interactive dashboard.
// Two things happen here, kept deliberately separate:
//   1. Triggering runs via mpc_api.py (http://127.0.0.1:8001) and polling job status.
//   2. Viewing results by reading the consolidated_*.csv files directly off disk
//      (same fetch()+PapaParse pattern storage.html/connectivity.html already use) -
//      the results viewer never goes through the API, it just reads whatever CSVs
//      are sitting in the chosen output folder.

const API_BASE = "http://127.0.0.1:8001";
const DATA_ROOT = "../mpc-pipeline"; // relative to this page

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------
// Run triggering + polling
// ---------------------------------------------------------------------

const activeJobIdByPrefix = {};   // "daily"/"season" -> current job_id, for the Stop button

function setJobStatus(prefix, { text, percent, cls, errorLog, logTail }) {
  const box = el(`${prefix}-status`);
  const textEl = el(`${prefix}-status-text`);
  const fill = el(`${prefix}-progress-fill`);
  const spinner = el(`${prefix}-spinner`);

  box.classList.add("visible");
  box.classList.remove("status-done", "status-error", "status-stopped");
  if (cls) box.classList.add(cls);

  textEl.textContent = text;
  if (percent !== undefined) fill.style.width = `${Math.max(4, percent)}%`;
  const settled = cls === "status-done" || cls === "status-error" || cls === "status-stopped";
  spinner.style.display = settled ? "none" : "inline-block";

  let logEl = box.querySelector("pre.error-log");
  if (errorLog) {
    if (!logEl) {
      logEl = document.createElement("pre");
      logEl.className = "error-log";
      box.appendChild(logEl);
    }
    logEl.textContent = errorLog;
  } else if (logEl) {
    logEl.remove();
  }

  if (logTail !== undefined) {
    const consoleEl = el(`${prefix}-console`);
    if (consoleEl) {
      consoleEl.textContent = logTail || "(no output yet)";
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
  }

  el(`${prefix}-stop-btn`).style.display = settled ? "none" : "inline-block";
}

function renderSummary(prefix, result) {
  const box = el(`${prefix}-summary`);
  if (!box) return;
  if (!result) {
    box.innerHTML = "";
    return;
  }
  const w = result.weights || {};
  const rows = (result.top_releases || [])
    .map((r) => `<tr><td>${r.tank_id}</td><td>${r.release_m3.toLocaleString()} m&sup3;</td></tr>`)
    .join("");
  box.innerHTML = `
    <div class="stat-row">
      <span><span class="k">Feasible:</span> <span class="v">${result.feasible}</span></span>
      <span><span class="k">P(drought):</span> <span class="v">${fmt(result.p_drought, 3)}</span></span>
      <span><span class="k">P(overflow):</span> <span class="v">${fmt(result.p_overflow, 3)}</span></span>
      <span><span class="k">C3 tier:</span> <span class="v">${result.c3_tier ?? "-"}</span></span>
      <span><span class="k">Total release:</span> <span class="v">${(result.total_release_m3 ?? 0).toLocaleString()} m&sup3; across ${result.n_tanks ?? "?"} tanks</span></span>
    </div>
    <div class="stat-row">
      <span><span class="k">Weights:</span> <span class="v">shortage ${fmt(w.shortage, 3)} &middot; overflow ${fmt(w.overflow, 3)} &middot; equity ${fmt(w.equity, 3)} &middot; loss ${fmt(w.loss, 3)}</span></span>
    </div>
    ${rows ? `<table><thead><tr><th>Top releases</th><th></th></tr></thead><tbody>${rows}</tbody></table>` : ""}
  `;
}

async function pollJob(prefix, jobId, { onDone } = {}) {
  const poll = async () => {
    let res;
    try {
      res = await fetch(`${API_BASE}/api/status?job=${encodeURIComponent(jobId)}`);
    } catch (err) {
      setJobStatus(prefix, {
        text: `Cannot reach the Module 4 API at ${API_BASE} (${err.message}). Is mpc_api.py running?`,
        cls: "status-error",
      });
      return;
    }
    const job = await res.json();

    if (job.status === "running") {
      let text = "Running...";
      let percent = 15;
      if (job.kind === "season" && job.day && job.total_days) {
        text = `Running - day ${job.day} / ${job.total_days}${job.last_feasible === false ? " (infeasible)" : ""}`;
        percent = Math.min(95, (job.day / job.total_days) * 100);
      }
      setJobStatus(prefix, { text, percent, logTail: job.log_tail });
      setTimeout(poll, 1200);
      return;
    }

    activeJobIdByPrefix[prefix] = null;

    if (job.status === "done") {
      const feasibleNote = job.result ? ` - feasible: ${job.result.feasible}` : "";
      setJobStatus(prefix, {
        text: `Done${feasibleNote}`,
        percent: 100,
        cls: "status-done",
        logTail: job.log_tail,
      });
      renderSummary(prefix, job.result);
      if (onDone) onDone(job);
      return;
    }

    if (job.status === "stopped") {
      const cleanupNote = job.cleaned_up ? " - incomplete output removed" : "";
      setJobStatus(prefix, {
        text: `Stopped${cleanupNote}`,
        percent: 100,
        cls: "status-stopped",
        logTail: job.log_tail,
      });
      return;
    }

    // error
    setJobStatus(prefix, {
      text: "Run failed - see log below",
      percent: 100,
      cls: "status-error",
      errorLog: job.error || "(no details captured)",
      logTail: job.log_tail,
    });
  };
  poll();
}

function wireStopButton(prefix) {
  el(`${prefix}-stop-btn`).addEventListener("click", async () => {
    const jobId = activeJobIdByPrefix[prefix];
    if (!jobId) return;
    el(`${prefix}-stop-btn`).disabled = true;
    try {
      await fetch(`${API_BASE}/api/stop?job=${encodeURIComponent(jobId)}`);
    } catch (err) {
      setJobStatus(prefix, { text: `Could not reach the API to stop the run (${err.message})`, cls: "status-error" });
    }
    el(`${prefix}-stop-btn`).disabled = false;
    // The in-flight poll() from pollJob() will pick up the "stopped" status
    // on its next tick and update the UI - no need to duplicate that here.
  });
}

function wireDailyRun() {
  el("daily-run-btn").addEventListener("click", async () => {
    const date = el("daily-date").value;
    if (!date) return;
    const fastTest = el("daily-fast-test").checked;
    el("daily-run-btn").disabled = true;
    renderSummary("daily", null);
    setJobStatus("daily", { text: "Starting...", percent: 5 });

    const url = new URL(`${API_BASE}/api/run/daily`);
    url.searchParams.set("date", date);
    if (fastTest) url.searchParams.set("fast_test", "1");

    let res;
    try {
      res = await fetch(url);
    } catch (err) {
      setJobStatus("daily", { text: `Cannot reach the API (${err.message}). Is mpc_api.py running?`, cls: "status-error" });
      el("daily-run-btn").disabled = false;
      return;
    }
    const payload = await res.json();
    el("daily-run-btn").disabled = false;
    if (!res.ok) {
      setJobStatus("daily", { text: payload.error || "Could not start run", cls: "status-error" });
      return;
    }
    activeJobIdByPrefix.daily = payload.job_id;
    pollJob("daily", payload.job_id, {
      onDone: () => {
        el("data-source").value = "outputs";
        loadAllTabs("outputs");
      },
    });
  });
}

function wireSeasonRun() {
  el("season-run-btn").addEventListener("click", async () => {
    const start = el("season-start").value;
    const end = el("season-end").value;
    if (!start || !end) return;
    el("season-run-btn").disabled = true;
    renderSummary("season", null);
    setJobStatus("season", { text: "Starting...", percent: 5 });

    const url = new URL(`${API_BASE}/api/run/season`);
    url.searchParams.set("start", start);
    url.searchParams.set("end", end);
    url.searchParams.set("season", el("season-season").value);
    url.searchParams.set("duration", el("season-duration").value);
    url.searchParams.set("forecast", el("season-forecast").value);
    if (el("season-output").value.trim()) url.searchParams.set("output", el("season-output").value.trim());
    if (el("season-fast-test").checked) url.searchParams.set("fast_test", "1");

    let res;
    try {
      res = await fetch(url);
    } catch (err) {
      setJobStatus("season", { text: `Cannot reach the API (${err.message}). Is mpc_api.py running?`, cls: "status-error" });
      el("season-run-btn").disabled = false;
      return;
    }
    const payload = await res.json();
    el("season-run-btn").disabled = false;
    if (!res.ok) {
      setJobStatus("season", { text: payload.error || "Could not start run", cls: "status-error" });
      return;
    }
    activeJobIdByPrefix.season = payload.job_id;
    pollJob("season", payload.job_id, {
      onDone: (job) => {
        el("data-source").value = job.output_dir;
        loadAllTabs(job.output_dir);
      },
    });
  });
}

// ---------------------------------------------------------------------
// Results viewer - generic table for the 3 consolidated CSVs
// ---------------------------------------------------------------------

function fmt(v, digits) {
  if (v === null || v === undefined || v === "") return "";
  if (digits !== undefined && typeof v === "number") return v.toFixed(digits);
  return v;
}

function createTableView({ prefix, csvFile, colCount, renderRow, filterFn, tankField, extraSelectField }) {
  let allRows = [];
  let filteredRows = [];
  let currentPage = 1;
  let pageSize = 50;

  function uniqueSorted(field) {
    return [...new Set(allRows.map((r) => r[field]).filter((v) => v !== null && v !== undefined && v !== ""))].sort();
  }

  function fillSelect(id, values, placeholder) {
    const sel = el(id);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = `<option value="">${placeholder}</option>`;
    for (const v of values) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    }
    sel.value = current;
  }

  function applyFilters() {
    filteredRows = allRows.filter((r) => filterFn(r));
    currentPage = 1;
    render();
  }

  function render() {
    const total = filteredRows.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    currentPage = Math.min(currentPage, totalPages);
    const start = (currentPage - 1) * pageSize;
    const pageRows = filteredRows.slice(start, start + pageSize);

    const body = el(`${prefix}-body`);
    body.innerHTML = pageRows.length
      ? pageRows.map(renderRow).join("")
      : `<tr class="empty-row"><td colspan="${colCount}">No rows match the current filters</td></tr>`;

    el(`${prefix}-info`).textContent =
      total === 0 ? "0 rows" : `Showing ${start + 1}-${Math.min(start + pageSize, total)} of ${total} rows`;
    el(`${prefix}-page-indicator`).textContent = `Page ${currentPage} of ${totalPages}`;
  }

  async function load(baseUrl) {
    const path = `${baseUrl}/${csvFile}`;
    el(`${prefix}-info`).textContent = "Loading...";
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
      allRows = parsed.data;
      if (tankField) fillSelect(`${prefix}-tank`, uniqueSorted(tankField), "All tanks");
      applyFilters();
    } catch (err) {
      allRows = [];
      el(`${prefix}-info`).textContent =
        `Could not load ${path} (${err.message}). Run something above first, or check the data source folder name.`;
      el(`${prefix}-body`).innerHTML = `<tr class="empty-row"><td colspan="${colCount}">No data loaded</td></tr>`;
    }
  }

  document.querySelectorAll(`[data-clear="${prefix}"]`).forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll(`#panel-${prefix} .filter-group input, #panel-${prefix} .filter-group select`)
        .forEach((f) => (f.value = ""));
      applyFilters();
    })
  );
  document.querySelectorAll(`[data-download="${prefix}"]`).forEach((btn) =>
    btn.addEventListener("click", () => {
      const csv = Papa.unparse(filteredRows);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${csvFile.replace(".csv", "")}_filtered.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    })
  );
  document.querySelectorAll(`[data-prev="${prefix}"]`).forEach((btn) =>
    btn.addEventListener("click", () => { if (currentPage > 1) { currentPage--; render(); } })
  );
  document.querySelectorAll(`[data-next="${prefix}"]`).forEach((btn) =>
    btn.addEventListener("click", () => { currentPage++; render(); })
  );
  document.querySelectorAll(`[data-page-size="${prefix}"]`).forEach((sel) =>
    sel.addEventListener("change", (e) => { pageSize = parseInt(e.target.value, 10); currentPage = 1; render(); })
  );
  document.querySelectorAll(`#panel-${prefix} .filter-group input, #panel-${prefix} .filter-group select`)
    .forEach((f) => f.addEventListener("change", applyFilters));

  return { load };
}

const decisionsView = createTableView({
  prefix: "decisions",
  csvFile: "consolidated_mpc_decisions.csv",
  colCount: 3,
  tankField: "tank_id",
  filterFn: (r) => {
    const tank = el("decisions-tank").value;
    const from = el("decisions-from").value;
    const to = el("decisions-to").value;
    if (tank && r.tank_id !== tank) return false;
    if (from && String(r.date) < from) return false;
    if (to && String(r.date) > to) return false;
    return true;
  },
  renderRow: (r) => `<tr><td>${fmt(r.date)}</td><td>${fmt(r.tank_id)}</td><td>${fmt(r.release_m3, 2)}</td></tr>`,
});

const weightsView = createTableView({
  prefix: "weights",
  csvFile: "consolidated_topsis_weights_log.csv",
  colCount: 11,
  filterFn: (r) => {
    const feas = el("weights-feasible").value;
    const from = el("weights-from").value;
    const to = el("weights-to").value;
    if (feas && String(r.feasible) !== feas) return false;
    if (from && String(r.date) < from) return false;
    if (to && String(r.date) > to) return false;
    return true;
  },
  renderRow: (r) => {
    const cls = String(r.feasible) === "True" ? "status-true" : "status-false";
    return `<tr>
      <td>${fmt(r.date)}</td><td>${fmt(r.day)}</td>
      <td>${fmt(r.p_drought, 3)}</td><td>${fmt(r.p_overflow, 3)}</td>
      <td>${fmt(r.w_shortage, 3)}</td><td>${fmt(r.w_overflow, 3)}</td>
      <td>${fmt(r.w_equity, 3)}</td><td>${fmt(r.w_loss, 3)}</td>
      <td class="${cls}">${fmt(r.feasible)}</td><td>${fmt(r.c3_tier)}</td>
      <td>${fmt(r.reserve_deficit_m3, 1)}</td>
    </tr>`;
  },
});

const crosscheckView = createTableView({
  prefix: "crosscheck",
  csvFile: "consolidated_module3_crosscheck.csv",
  colCount: 7,
  tankField: "tank_id",
  filterFn: (r) => {
    const tank = el("crosscheck-tank").value;
    const from = el("crosscheck-from").value;
    const to = el("crosscheck-to").value;
    if (tank && r.tank_id !== tank) return false;
    if (from && String(r.date) < from) return false;
    if (to && String(r.date) > to) return false;
    return true;
  },
  renderRow: (r) => `<tr>
    <td>${fmt(r.date)}</td><td>${fmt(r.tank_id)}</td><td>${fmt(r.day)}</td>
    <td>${fmt(r.m4_volume_ratio, 4)}</td><td>${fmt(r.m3_volume_ratio, 4)}</td>
    <td>${fmt(r.ratio_difference, 4)}</td><td>${fmt(r.abs_ratio_difference, 4)}</td>
  </tr>`,
});

function loadAllTabs(folder) {
  const baseUrl = `${DATA_ROOT}/${folder}`;
  el("data-source-hint").textContent = `Loading from ${baseUrl}/`;
  decisionsView.load(baseUrl);
  weightsView.load(baseUrl);
  crosscheckView.load(baseUrl);
}

function wireDataSource() {
  el("data-source-load").addEventListener("click", () => {
    const folder = el("data-source").value.trim() || "outputs";
    loadAllTabs(folder);
  });
}

function wireTabs() {
  const steps = document.querySelectorAll(".step-item");
  steps.forEach((step) => {
    step.addEventListener("click", () => {
      steps.forEach((s) => s.classList.toggle("active", s === step));
      document.querySelectorAll(".view-panel").forEach((panel) => {
        panel.hidden = panel.id !== `panel-${step.dataset.target}`;
      });
    });
  });
}

wireDailyRun();
wireSeasonRun();
wireStopButton("daily");
wireStopButton("season");
wireDataSource();
wireTabs();
loadAllTabs("outputs");
