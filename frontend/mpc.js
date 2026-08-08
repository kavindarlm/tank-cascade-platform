// Module 4 interactive dashboard.
// Three things happen here, kept deliberately separate:
//   1. Triggering runs via Module 4's MPC API (mounted in the root main.py
//      alongside Module 3's forecast API - same origin as this page) and
//      polling job status.
//   2. Loading the consolidated_*.csv files off disk ONCE into `store` (same
//      fetch()+PapaParse pattern storage.html/connectivity.html already use) -
//      the results viewer never goes through the API, it just reads whatever
//      CSVs are sitting in the chosen output folder.
//   3. Rendering that one store into plain-language views. The pipeline's
//      vocabulary (TOPSIS, C3, w_shortage, volume ratio) is deliberately kept
//      out of the headline reading path: each tab answers a question, and the
//      raw table that backs it stays one click away for auditing.

const API_BASE = ""; // same origin - main.py serves this page and the API together
const DATA_ROOT = "../Module4/mpc-pipeline"; // relative to this page

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------

const store = {
  decisions: [],     // {date, tank_id, release_m3, demand_m3}
  weights: [],       // {date, p_drought, p_overflow, w_*, feasible, c3_tier, ...}
  crosscheck: [],    // {date, tank_id, day, m4_volume_ratio, m3_volume_ratio, ...}
  dates: [],         // sorted unique dates present in the loaded run
  selectedDate: "",  // drives every tab; set by the one date bar at the top
  deliveryFilter: "all",
  agreementFilter: "all",
};

// Fallback only. The live values come from /api/config (mpc_api.py's
// _ui_config_payload) so that retuning module4/config.py can never leave the
// page quoting a threshold the pipeline no longer uses - the Forecast
// agreement tab's tolerance in particular. base_weights is no longer rendered
// anywhere (the weight cards show the applied % on its own), but is kept as
// the sanity check that /api/config returned a real payload.
let uiConfig = {
  base_weights: { shortage: 0.35, overflow: 0.25, equity: 0.2, loss: 0.2 },
  crosscheck_divergence_threshold: 0.15,
  horizon_days: 7,
};

// The four optimiser objectives, in the order the CSV columns appear, with the
// plain-language name each one is shown under. objectives.py defines them as
// costs (lower is better); users think in terms of the goal, not the cost, so
// the label names the goal.
const GOALS = [
  { key: "shortage", col: "w_shortage", name: "Meeting demand",      desc: "Avoid leaving fields without water" },
  { key: "overflow", col: "w_overflow", name: "Preventing overflow", desc: "Avoid water spilling over the tank bund" },
  { key: "equity",   col: "w_equity",   name: "Fair sharing",        desc: "Spread any shortage evenly between tanks" },
  { key: "loss",     col: "w_loss",     name: "Avoiding waste",      desc: "Don't release more water than is needed" },
];

const TIER_LABEL = {
  served: "Fully served",
  partial: "Partly served",
  critical: "Critically short",
  nodata: "No data",
};

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const n0 = (v) => Math.round(Number(v) || 0).toLocaleString();
const tankLabel = (id) => String(id ?? "").replace(/_/g, " ");

function prettyDate(iso) {
  const d = new Date(`${iso}T00:00:00`);
  if (isNaN(d)) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

// Mutually exclusive buckets, so the counts always sum to the tank total.
// Anything >= 90% (including over 100%) counts as "served" - a tank that
// got more than it asked for still had its demand fully met, which is what
// this status is answering ("was this tank's need satisfied?"), not "did
// the release exactly equal demand?".
function classify(pct) {
  if (pct === null || pct === undefined || !isFinite(pct)) return "nodata";
  if (pct >= 90) return "served";
  if (pct >= 50) return "partial";
  return "critical";
}

function satisfactionPctOf(row) {
  const demand = Number(row.demand_m3);
  if (!isFinite(demand) || demand <= 0) return null;
  return (Number(row.release_m3) / demand) * 100;
}

function decisionsForDate(date) {
  return store.decisions
    .filter((r) => String(r.date) === date && r.tank_id)
    .map((r) => {
      const pct = satisfactionPctOf(r);
      return {
        tank_id: r.tank_id,
        release: Number(r.release_m3) || 0,
        demand: Number(r.demand_m3),
        pct,
        tier: classify(pct),
      };
    });
}

function weightsForDate(date) {
  return store.weights.find((r) => String(r.date) === date) || null;
}

/** Network-level totals for one date. */
function summarise(rows) {
  const counts = { served: 0, partial: 0, critical: 0, nodata: 0 };
  let totalDemand = 0;
  let totalRelease = 0;
  let covered = 0; // min(release, demand) - demand actually met, ignoring overshoot

  rows.forEach((r) => {
    counts[r.tier] += 1;
    totalRelease += r.release;
    if (isFinite(r.demand) && r.demand > 0) {
      totalDemand += r.demand;
      covered += Math.min(r.release, r.demand);
    }
  });

  return {
    n: rows.length,
    counts,
    totalDemand,
    totalRelease,
    covered,
    // Coverage counts only water that met a real need. Using
    // totalRelease/totalDemand instead would let a tank given 3x its demand
    // paper over a neighbour that got nothing.
    coveragePct: totalDemand > 0 ? (covered / totalDemand) * 100 : null,
  };
}

// A network-wide "% of demand met" is a demand-weighted average, so one tank
// that is far larger than the rest can single-handedly decide the headline
// number even when every small "village" tank is going short - a genuinely
// misleading read of "is the network okay?". Flag tanks whose CAPACITY
// (s_max_m3, a fixed physical property, not something that shifts day to
// day like demand does) is a large multiple of the median tank's capacity.
// Threshold picked from this cascade's real spread: the biggest reservoir
// here sits at ~300x the median tank, the next-biggest step is only ~6x -
// 20x cleanly isolates a genuine outlier without catching an ordinarily
// bigger village tank.
const LARGE_RESERVOIR_MEDIAN_MULTIPLE = 20;

function largeReservoirIds(tanksById) {
  const caps = Object.values(tanksById).map((t) => t.s_max_m3).filter((v) => isFinite(v) && v > 0).sort((a, b) => a - b);
  if (!caps.length) return new Set();
  const median = caps[Math.floor(caps.length / 2)];
  if (!median) return new Set();
  return new Set(
    Object.values(tanksById)
      .filter((t) => t.s_max_m3 > median * LARGE_RESERVOIR_MEDIAN_MULTIPLE)
      .map((t) => t.tank_id)
  );
}

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
    .map((r) => `<tr><td>${esc(tankLabel(r.tank_id))}</td><td>${n0(r.release_m3)} m&sup3;</td></tr>`)
    .join("");
  box.innerHTML = `
    <div class="stat-row">
      <span><span class="k">Plan usable:</span> <span class="v">${result.feasible ? "yes" : "no"}</span></span>
      <span><span class="k">Drought risk:</span> <span class="v">${Math.round((result.p_drought ?? 0) * 100)}%</span></span>
      <span><span class="k">Overflow risk:</span> <span class="v">${Math.round((result.p_overflow ?? 0) * 100)}%</span></span>
      <span><span class="k">Total release:</span> <span class="v">${n0(result.total_release_m3)} m&sup3; across ${result.n_tanks ?? "?"} tanks</span></span>
    </div>
    <div class="stat-row">
      <span><span class="k">Priorities:</span> <span class="v">${GOALS.map(
        (g) => `${g.name} ${Math.round((w[g.key] ?? 0) * 100)}%`
      ).join(" &middot; ")}</span></span>
    </div>
    ${rows ? `<table><thead><tr><th>Largest releases</th><th></th></tr></thead><tbody>${rows}</tbody></table>` : ""}
  `;
}

async function pollJob(prefix, jobId, { onDone } = {}) {
  const poll = async () => {
    let job;
    try {
      const res = await fetch(`${API_BASE}/api/status?job=${encodeURIComponent(jobId)}`);
      job = await res.json();
    } catch (err) {
      setJobStatus(prefix, {
        text: `Cannot reach the Module 4 API (${err.message}). Is main.py running?`,
        cls: "status-error",
      });
      return;
    }

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
      const feasibleNote = job.result ? ` - plan usable: ${job.result.feasible ? "yes" : "no"}` : "";
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

    const url = new URL(`${API_BASE}/api/run/daily`, window.location.origin);
    url.searchParams.set("date", date);
    if (fastTest) url.searchParams.set("fast_test", "1");

    let payload;
    let ok;
    try {
      const res = await fetch(url);
      ok = res.ok;
      payload = await res.json();
    } catch (err) {
      setJobStatus("daily", { text: `Cannot reach the API (${err.message}). Is main.py running?`, cls: "status-error" });
      el("daily-run-btn").disabled = false;
      return;
    }
    el("daily-run-btn").disabled = false;
    if (!ok) {
      setJobStatus("daily", { text: payload.error || "Could not start run", cls: "status-error" });
      return;
    }
    activeJobIdByPrefix.daily = payload.job_id;
    pollJob("daily", payload.job_id, {
      onDone: () => {
        el("data-source").value = "outputs";
        // Land on the date that was just computed, not whatever was selected
        // before the run - otherwise a finished run appears to change nothing.
        loadAllTabs("outputs", { preferDate: date });
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

    const url = new URL(`${API_BASE}/api/run/season`, window.location.origin);
    url.searchParams.set("start", start);
    url.searchParams.set("end", end);
    url.searchParams.set("season", el("season-season").value);
    url.searchParams.set("duration", el("season-duration").value);
    url.searchParams.set("forecast", el("season-forecast").value);
    if (el("season-output").value.trim()) url.searchParams.set("output", el("season-output").value.trim());
    if (el("season-fast-test").checked) url.searchParams.set("fast_test", "1");

    let payload;
    let ok;
    try {
      const res = await fetch(url);
      ok = res.ok;
      payload = await res.json();
    } catch (err) {
      setJobStatus("season", { text: `Cannot reach the API (${err.message}). Is main.py running?`, cls: "status-error" });
      el("season-run-btn").disabled = false;
      return;
    }
    el("season-run-btn").disabled = false;
    if (!ok) {
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
// Raw data tables - the auditable escape hatch behind each summary view.
// Rows are handed in from `store` (fetched once in loadAllTabs) rather than
// re-fetched here, and these keep their own independent date filters so the
// full run stays reachable regardless of the date bar at the top.
// ---------------------------------------------------------------------

function fmt(v, digits) {
  if (v === null || v === undefined || v === "") return "";
  if (digits !== undefined && typeof v === "number") return v.toFixed(digits);
  return v;
}

function createTableView({ prefix, csvFile, colCount, renderRow, filterFn, tankField }) {
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

  function setRows(rows, errorMessage) {
    allRows = rows || [];
    if (errorMessage) {
      el(`${prefix}-info`).textContent = errorMessage;
      el(`${prefix}-body`).innerHTML = `<tr class="empty-row"><td colspan="${colCount}">No data loaded</td></tr>`;
      return;
    }
    if (tankField) fillSelect(`${prefix}-tank`, uniqueSorted(tankField), "All tanks");
    applyFilters();
  }

  document.querySelectorAll(`[data-clear="${prefix}"]`).forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll(`#raw-${prefix} .filter-group input, #raw-${prefix} .filter-group select`)
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
  document.querySelectorAll(`#raw-${prefix} .filter-group input, #raw-${prefix} .filter-group select`)
    .forEach((f) => f.addEventListener("change", applyFilters));

  return { setRows };
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
  renderRow: (r) => `<tr><td>${fmt(r.date)}</td><td>${esc(r.tank_id)}</td><td>${fmt(r.release_m3, 2)}</td></tr>`,
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
    <td>${fmt(r.date)}</td><td>${esc(r.tank_id)}</td><td>${fmt(r.day)}</td>
    <td>${fmt(r.m4_volume_ratio, 4)}</td><td>${fmt(r.m3_volume_ratio, 4)}</td>
    <td>${fmt(r.ratio_difference, 4)}</td><td>${fmt(r.abs_ratio_difference, 4)}</td>
  </tr>`,
});

// ---------------------------------------------------------------------
// Overview - the answer before the data
// ---------------------------------------------------------------------

function tile(value, label, sub, cls) {
  return `<div class="tile ${cls || ""}">
    <div class="tile-value">${value}</div>
    <div class="tile-label">${label}</div>
    ${sub ? `<div class="tile-sub">${sub}</div>` : ""}
  </div>`;
}

function renderOverview() {
  const tiles = el("overview-tiles");
  const verdict = el("overview-verdict");
  const scale = el("overview-scale");
  const dist = el("overview-dist");
  const concerns = el("overview-concerns");

  const rows = decisionsForDate(store.selectedDate);
  if (!rows.length) {
    tiles.innerHTML = "";
    if (scale) scale.innerHTML = "";
    verdict.className = "verdict";
    verdict.innerHTML = store.dates.length
      ? "No release decisions recorded for this date."
      : "No results loaded yet. Run a day or a season above, or point the data source folder at an existing run and click Load.";
    dist.innerHTML = "";
    concerns.innerHTML = "";
    return;
  }

  const s = summarise(rows);
  const w = weightsForDate(store.selectedDate);
  const cov = s.coveragePct === null ? null : Math.round(s.coveragePct);
  const fullyServed = s.counts.served;
  const feasible = w ? String(w.feasible) === "True" : null;

  const covCls = cov === null ? "" : cov >= 90 ? "good" : cov >= 60 ? "warn" : "bad";
  const critCls = s.counts.critical === 0 ? "good" : s.counts.critical > s.n / 4 ? "bad" : "warn";

  tiles.innerHTML = [
    tile(
      cov === null ? "-" : `${cov}%`,
      `of water demand delivered <i class="info" data-tip="Water that met a real need, divided by total demand. Water released beyond a tank's demand is not counted here - it can't make up for a different tank going short.">i</i>`,
      `${n0(s.covered)} of ${n0(s.totalDemand)} m&sup3;`,
      covCls
    ),
    tile(`${fullyServed} <span style="font-size:15px;color:#8a8f9c">of ${s.n}</span>`,
      "tanks got enough water", "received 90% or more of their demand", fullyServed === s.n ? "good" : ""),
    tile(String(s.counts.critical), "tanks critically short", "received under half their demand", critCls),
    tile(
      feasible === null ? "-" : feasible ? "Usable" : "Unusable",
      `plan status <i class="info" data-tip="Usable means the chosen plan respected every physical limit: tank storage stayed inside its safe range, releases stayed within valve capacity, and the seasonal reserve floor held.">i</i>`,
      feasible === null ? "no record for this date" : feasible ? "all physical limits respected" : "a physical limit was breached",
      feasible === null ? "" : feasible ? "good" : "bad"
    ),
    tile(n0(s.totalRelease), "m&sup3; released in total", `across ${s.n} tanks`),
  ].join("");

  // Verdict sentence - the same numbers, read as prose.
  const parts = [];
  parts.push(`On <strong>${prettyDate(store.selectedDate)}</strong> the plan delivered <strong>${cov === null ? "an unknown share of" : cov + "% of"}</strong> the network's water demand.`);
  const clauses = [];
  if (s.counts.served) clauses.push(`<strong>${s.counts.served}</strong> ${s.counts.served === 1 ? "tank" : "tanks"} had ${s.counts.served === 1 ? "its" : "their"} demand fully met`);
  if (s.counts.partial) clauses.push(`<strong>${s.counts.partial}</strong> ${s.counts.partial === 1 ? "was" : "were"} partly served`);
  if (s.counts.critical) clauses.push(`<strong>${s.counts.critical}</strong> fell below half their requirement`);
  if (clauses.length) parts.push(`${clauses.join(", ")}.`);
  if (feasible === true) parts.push("All physical constraints were satisfied, so this plan is safe to apply.");
  else if (feasible === false) parts.push("<strong>A physical constraint was breached</strong> - review before applying this plan.");

  verdict.className = `verdict ${feasible === false || (cov !== null && cov < 60) ? "bad" : s.counts.critical ? "warn" : "good"}`;
  verdict.innerHTML = parts.join(" ");

  // Village tanks vs. the network-wide figure. A demand-weighted average is
  // dominated by whichever tank has the most demand - if one reservoir is
  // far bigger than the rest, the headline % mostly describes THAT tank,
  // not the small village tanks most people mean by "is the network okay?".
  // Only rendered when a genuine size outlier exists (largeReservoirIds()),
  // so this stays silent for a dataset without one.
  if (scale) {
    const tanksById = mapView.getTanksById ? mapView.getTanksById() : {};
    const largeIds = largeReservoirIds(tanksById);
    const largeRows = rows.filter((r) => largeIds.has(r.tank_id));
    if (largeRows.length) {
      const villageRows = rows.filter((r) => !largeIds.has(r.tank_id));
      const villageStats = summarise(villageRows);
      const vCov = villageStats.coveragePct === null ? null : Math.round(villageStats.coveragePct);
      const vCls = vCov === null ? "" : vCov >= 90 ? "good" : vCov >= 60 ? "warn" : "bad";
      const names = largeRows.map((r) => tankLabel(r.tank_id)).join(", ");
      const largeDemand = largeRows.reduce((a, r) => a + (isFinite(r.demand) && r.demand > 0 ? r.demand : 0), 0);
      const largeShare = s.totalDemand > 0 ? Math.round((largeDemand / s.totalDemand) * 100) : null;
      scale.innerHTML = `
        <div class="scale-box">
          <p class="sec-note" style="margin-bottom:12px;">
            <strong>${esc(names)}</strong> ${largeRows.length === 1 ? "is" : "are"} far larger than every other tank in
            this cascade${largeShare !== null ? ` - alone ${largeRows.length === 1 ? "it accounts" : "they account"} for
            <strong>${largeShare}%</strong> of total network demand` : ""}. That means the network-wide figure above
            mostly reflects ${largeRows.length === 1 ? "that one tank" : "those tanks"} - excluding
            ${largeRows.length === 1 ? "it" : "them"} shows how the smaller village tanks are actually doing.
          </p>
          <div class="scale-compare">
            <div class="scale-block ${covCls}">
              <div class="scale-value">${cov === null ? "-" : cov + "%"}</div>
              <div class="scale-label">Network-wide<br>(all ${s.n} tanks)</div>
            </div>
            <div class="scale-arrow">&rarr;</div>
            <div class="scale-block ${vCls}">
              <div class="scale-value">${vCov === null ? "-" : vCov + "%"}</div>
              <div class="scale-label">Village tanks only<br>(${villageStats.n} tanks, excluding ${esc(names)})</div>
            </div>
          </div>
        </div>`;
    } else {
      scale.innerHTML = "";
    }
  }

  // Distribution bar
  const order = ["served", "partial", "critical"];
  const segs = order
    .filter((k) => s.counts[k] > 0)
    .map((k) => `<div class="dist-seg ${k}" style="flex-grow:${s.counts[k]}" title="${TIER_LABEL[k]}: ${s.counts[k]} tanks">${s.counts[k]}</div>`)
    .join("");
  dist.innerHTML = `
    <div class="dist-bar">${segs || '<div class="dist-seg" style="flex-grow:1"></div>'}</div>
    <div class="dist-legend">
      ${order.map((k) => `<span><span class="swatch-sq ${k}"></span> ${TIER_LABEL[k]} - ${s.counts[k]}</span>`).join("")}
    </div>`;

  // Worst offenders, so "what do I do about it" has an answer on this screen.
  const worst = rows
    .filter((r) => r.tier === "critical" || r.tier === "partial")
    .sort((a, b) => a.pct - b.pct)
    .slice(0, 5);

  concerns.innerHTML = worst.length
    ? `<div class="delivery-list">
        <div class="drow head"><div>Tank</div><div>Demand met</div><div>Delivered / needed</div><div>%</div><div>Status</div></div>
        ${worst.map(deliveryRow).join("")}
      </div>
      ${s.counts.critical + s.counts.partial > worst.length
        ? `<p class="sec-note" style="margin-top:10px;">Showing the 5 worst of ${s.counts.critical + s.counts.partial} under-served tanks - see the Water delivery tab for the full list.</p>`
        : ""}`
    : `<div class="delivery-list"><div class="empty-state">Every tank received at least 90% of its demand on this date.</div></div>`;
}

// ---------------------------------------------------------------------
// Water delivery - ranked, worst first
// ---------------------------------------------------------------------

function deliveryRow(r) {
  const pctText = r.pct === null ? "-" : `${Math.round(r.pct)}%`;
  const fillPct = r.pct === null ? 0 : Math.min(100, r.pct);
  const needed = isFinite(r.demand) && r.demand > 0 ? `${n0(r.release)} / ${n0(r.demand)} m&sup3;` : `${n0(r.release)} m&sup3; (no demand recorded)`;
  return `<div class="drow">
    <div class="dname" title="${esc(tankLabel(r.tank_id))}">${esc(tankLabel(r.tank_id))}</div>
    <div class="meter"><div class="meter-fill ${r.tier}" style="width:${fillPct}%"></div></div>
    <div class="dnums">${needed}</div>
    <div class="dpct ${r.tier}">${pctText}</div>
    <div><span class="pill ${r.tier}">${TIER_LABEL[r.tier]}</span></div>
  </div>`;
}

function renderDelivery() {
  const chips = el("delivery-chips");
  const list = el("delivery-list");
  const rows = decisionsForDate(store.selectedDate);

  if (!rows.length) {
    chips.innerHTML = "";
    list.innerHTML = `<div class="empty-state">No release decisions recorded for this date.</div>`;
    return;
  }

  const counts = { all: rows.length, critical: 0, partial: 0, served: 0 };
  rows.forEach((r) => { if (counts[r.tier] !== undefined) counts[r.tier] += 1; });

  const chipDefs = [
    { key: "all", label: "All tanks" },
    { key: "critical", label: TIER_LABEL.critical },
    { key: "partial", label: TIER_LABEL.partial },
    { key: "served", label: TIER_LABEL.served },
  ];
  chips.innerHTML = chipDefs
    .map((c) => `<button type="button" class="chip ${store.deliveryFilter === c.key ? "active" : ""}" data-filter="${c.key}">${c.label} (${counts[c.key]})</button>`)
    .join("");
  chips.querySelectorAll(".chip").forEach((btn) =>
    btn.addEventListener("click", () => {
      store.deliveryFilter = btn.dataset.filter;
      renderDelivery();
    })
  );

  // Worst first - that is the order someone acts on. nodata sinks to the end.
  const shown = rows
    .filter((r) => store.deliveryFilter === "all" || r.tier === store.deliveryFilter)
    .sort((a, b) => {
      if (a.pct === null) return 1;
      if (b.pct === null) return -1;
      return a.pct - b.pct;
    });

  list.innerHTML = shown.length
    ? `<div class="drow head"><div>Tank</div><div>Demand met</div><div>Delivered / needed</div><div>%</div><div>Status</div></div>
       ${shown.map(deliveryRow).join("")}`
    : `<div class="empty-state">No tanks in this category on ${esc(prettyDate(store.selectedDate))}.</div>`;
}

// ---------------------------------------------------------------------
// Why this plan - the four weights, translated into goals
// ---------------------------------------------------------------------

function renderWhy() {
  const weightsBox = el("why-weights");
  const reasonBox = el("why-reason");
  const statusBox = el("why-status");

  const w = weightsForDate(store.selectedDate);
  if (!w) {
    weightsBox.innerHTML = `<div class="delivery-list"><div class="empty-state">No decision record for this date.</div></div>`;
    reasonBox.innerHTML = "";
    statusBox.innerHTML = "";
    return;
  }

  const vals = GOALS.map((g) => ({ ...g, value: Number(w[g.col]) || 0 }));
  const total = vals.reduce((a, b) => a + b.value, 0) || 1;

  weightsBox.innerHTML = `
    <div class="weight-bar">
      ${vals.map((v) => {
        const pct = (v.value / total) * 100;
        return `<div class="wseg ${v.key}" style="flex-grow:${v.value}" title="${v.name}: ${pct.toFixed(0)}%">${pct >= 12 ? `${pct.toFixed(0)}%` : ""}</div>`;
      }).join("")}
    </div>
    <div class="weight-legend">
      ${vals.map((v) => {
        const pct = (v.value / total) * 100;
        return `<div class="wlegend-item">
          <span class="swatch-sq ${v.key}"></span>
          <div>
            <div class="wl-name">${v.name}</div>
            <div class="wl-val">${pct.toFixed(0)}%</div>
            <div class="wl-desc">${v.desc}</div>
          </div>
        </div>`;
      }).join("")}
    </div>`;

  // The risk forecast that drove the priorities above.
  const pd = Number(w.p_drought) || 0;
  const po = Number(w.p_overflow) || 0;

  reasonBox.innerHTML = `
    <div class="risk-row">
      <div class="risk-card">
        <div class="rk-name">Chance of drought <i class="info" data-tip="Module 3's forecast probability that the network moves into drought conditions. It is the only place Module 3's risk enters Module 4.">i</i></div>
        <div class="rk-val">${Math.round(pd * 100)}%</div>
        <div class="risk-track"><div class="risk-fill drought" style="width:${Math.min(100, pd * 100)}%"></div></div>
      </div>
      <div class="risk-card">
        <div class="rk-name">Chance of overflow <i class="info" data-tip="Module 3's forecast probability that tanks spill over their bunds. High values push the plan to release water earlier.">i</i></div>
        <div class="rk-val">${Math.round(po * 100)}%</div>
        <div class="risk-track"><div class="risk-fill overflow" style="width:${Math.min(100, po * 100)}%"></div></div>
      </div>
    </div>`;

  // Plan status, in words rather than flag names.
  const feasible = String(w.feasible) === "True";
  const tier = String(w.c3_tier || "").toLowerCase();
  const tierText = tier === "hard" ? "Strict" : tier === "relaxed" ? "Relaxed" : (w.c3_tier || "-");
  const tierExplain = tier === "hard"
    ? "Tanks were held to the full seasonal reserve level."
    : tier === "relaxed"
      ? "Tanks already below the seasonal reserve were only required not to fall further than doing nothing would have."
      : "No reserve rule recorded for this date.";
  const deficit = w.reserve_deficit_m3;
  const hasDeficit = deficit !== null && deficit !== undefined && deficit !== "" && Number(deficit) > 0;

  statusBox.innerHTML = `<div class="kv-grid">
    <div class="kv-card">
      <div class="kv-k">Plan usable</div>
      <div class="kv-v" style="color:${feasible ? "#059669" : "#dc2626"}">${feasible ? "Yes - all limits respected" : "No - a limit was breached"}</div>
      <div class="wl-desc">${feasible
        ? "Storage stayed in range, releases stayed within valve limits, and the reserve floor held."
        : "At least one physical constraint could not be satisfied. Treat these releases as advisory."}</div>
    </div>
    <div class="kv-card">
      <div class="kv-k">Reserve rule <i class="info" data-tip="Constraint C3: the seasonal reserve each tank must keep back. It relaxes to a do-no-harm floor when a tank already starts below that level, so the optimiser is never handed an impossible problem.">i</i></div>
      <div class="kv-v">${esc(tierText)}</div>
      <div class="wl-desc">${tierExplain}</div>
    </div>
    <div class="kv-card">
      <div class="kv-k">Reserve shortfall</div>
      <div class="kv-v" style="color:${hasDeficit ? "#ca8a04" : "#059669"}">${hasDeficit ? `${n0(deficit)} m&sup3;` : "None"}</div>
      <div class="wl-desc">${hasDeficit
        ? "Storage sits below the absolute seasonal reserve by this volume."
        : "No tank ended below its absolute seasonal reserve."}</div>
    </div>
  </div>`;
}

// ---------------------------------------------------------------------
// Forecast agreement - a verdict, not 600 rows
// ---------------------------------------------------------------------

/**
 * Two-line sparkline of Module 4 vs Module 3 volume ratio over the horizon.
 *
 * The Y-axis floor is never narrower than the divergence tolerance band. Pure
 * auto-scaling (fit tightly to each tank's own min/max) made an in-tolerance
 * tank's few-percent noise stretch to fill the whole chart height, looking
 * exactly as dramatic as a tank genuinely outside tolerance - there was no
 * way to tell "meaningless wobble" from "real divergence" by shape alone.
 * Flooring the range at +/-2x threshold means a tank that's actually fine
 * renders as a near-flat line (the honest picture), while a real outlier
 * still expands the scale and visibly blows past it.
 */
function sparkline(days, threshold) {
  const W = 150, H = 34, PAD = 3;
  const vals = days.flatMap((d) => [d.m4, d.m3]).filter((v) => isFinite(v));
  if (vals.length < 2) return "";
  const band = Math.max(Number(threshold) || 0, 0.01) * 2;
  let lo = Math.min(...vals, 1 - band), hi = Math.max(...vals, 1 + band);
  if (hi - lo < 1e-6) { lo -= 0.05; hi += 0.05; }
  const x = (i) => PAD + (i / Math.max(1, days.length - 1)) * (W - 2 * PAD);
  const y = (v) => PAD + (1 - (v - lo) / (hi - lo)) * (H - 2 * PAD);
  const path = (key) => days.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(d[key]).toFixed(1)}`).join(" ");
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Module 4 versus Module 3 projected water level over ${days.length} days">
    <line x1="${PAD}" y1="${y(1).toFixed(1)}" x2="${W - PAD}" y2="${y(1).toFixed(1)}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2 2"/>
    <path d="${path("m3")}" fill="none" stroke="#64748b" stroke-width="1.6" stroke-dasharray="3 2"/>
    <path d="${path("m4")}" fill="none" stroke="#2b6cb0" stroke-width="1.8"/>
  </svg>`;
}

function renderAgreement() {
  const summaryBox = el("agreement-summary");
  const chips = el("agreement-chips");
  const listBox = el("agreement-list");

  const rows = store.crosscheck.filter((r) => String(r.date) === store.selectedDate && r.tank_id);
  if (!rows.length) {
    summaryBox.innerHTML = `<div class="verdict">No cross-check data recorded for this date.</div>`;
    chips.innerHTML = "";
    listBox.innerHTML = "";
    return;
  }

  const threshold = uiConfig.crosscheck_divergence_threshold;
  const byTank = new Map();
  rows.forEach((r) => {
    if (!byTank.has(r.tank_id)) byTank.set(r.tank_id, []);
    byTank.get(r.tank_id).push(r);
  });

  const tanks = [...byTank.entries()].map(([tank_id, rs]) => {
    const diffs = rs.map((r) => Math.abs(Number(r.abs_ratio_difference) || 0));
    const mean = diffs.reduce((a, b) => a + b, 0) / (diffs.length || 1);
    const days = rs
      .slice()
      .sort((a, b) => Number(a.day) - Number(b.day))
      .map((r) => ({ m4: Number(r.m4_volume_ratio), m3: Number(r.m3_volume_ratio) }));
    return { tank_id, mean, max: Math.max(...diffs), days };
  }).sort((a, b) => b.mean - a.mean);

  const overall = tanks.reduce((a, t) => a + t.mean, 0) / (tanks.length || 1);
  const flagged = tanks.filter((t) => t.mean > threshold);
  const ok = flagged.length === 0;

  summaryBox.innerHTML = `
    <div class="stat-tiles">
      ${tile(overall.toFixed(3),
        `average disagreement <i class="info" data-tip="Mean absolute difference between the two modules' projected water level, expressed as a ratio of each tank's starting volume. 0 means perfect agreement.">i</i>`,
        `tolerance is ${threshold}`, overall <= threshold ? "good" : "bad")}
      ${tile(String(tanks.length), "tanks compared", `over ${uiConfig.horizon_days} forecast days`)}
    </div>
    <div class="verdict ${ok ? "good" : "warn"}">
      ${ok
        ? `<strong>Good agreement.</strong> Module 4's simulation and Module 3's forecast track each other closely for all ${tanks.length} tanks on ${prettyDate(store.selectedDate)} - average difference ${overall.toFixed(3)}, well inside the ${threshold} tolerance. The release plan is consistent with the independent forecast.`
        : `<strong>${flagged.length} of ${tanks.length} tanks diverge.</strong> For these tanks Module 4 expects the water level to move differently from Module 3's forecast by more than the ${threshold} tolerance. That does not make either wrong, but their release plans are the ones worth reviewing first.`}
    </div>`;

  const chipDefs = [
    { key: "all", label: "All tanks", count: tanks.length },
    { key: "flagged", label: "Disagreeing", count: flagged.length },
  ];
  chips.innerHTML = chipDefs
    .map((c) => `<button type="button" class="chip ${store.agreementFilter === c.key ? "active" : ""}" data-filter="${c.key}">${c.label} (${c.count})</button>`)
    .join("");
  chips.querySelectorAll(".chip").forEach((btn) =>
    btn.addEventListener("click", () => {
      store.agreementFilter = btn.dataset.filter;
      renderAgreement();
    })
  );

  // "all" shows every compared tank, worst-first (tanks is already sorted
  // that way); "flagged" narrows to just the ones outside tolerance.
  const shown = store.agreementFilter === "flagged" ? flagged : tanks;
  listBox.innerHTML = shown.length
    ? `<div class="agree-list">
        <div class="arow head">
          <div>Tank</div><div>Avg difference</div><div>Projected water level (7 days)</div><div>Status</div>
        </div>
        ${shown.map((t) => `<div class="arow">
          <div class="aname">${esc(tankLabel(t.tank_id))}</div>
          <div class="aval">${t.mean.toFixed(3)} <span style="color:#8a8f9c">(max ${t.max.toFixed(3)})</span></div>
          <div>${sparkline(t.days, threshold)}</div>
          <div><span class="pill ${t.mean > threshold ? "partial" : "served"}">${t.mean > threshold ? "Diverging" : "In tolerance"}</span></div>
        </div>`).join("")}
      </div>
      <div class="spark-legend">
        <span><span class="ln" style="border-color:#2b6cb0"></span>Module 4 (this plan)</span>
        <span><span class="ln" style="border-color:#64748b;border-top-style:dashed"></span>Module 3 (forecast)</span>
        <span><span class="ln" style="border-color:#cbd5e1;border-top-style:dashed"></span>Starting level</span>
      </div>`
    : `<div class="agree-list"><div class="empty-state">No tanks are outside tolerance for this date.</div></div>`;
}

// ---------------------------------------------------------------------
// Loading + wiring
// ---------------------------------------------------------------------

async function fetchCsv(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const parsed = Papa.parse(await res.text(), { header: true, dynamicTyping: true, skipEmptyLines: true });
    return { rows: parsed.data.filter((r) => r && r.date), error: null };
  } catch (err) {
    return { rows: [], error: `Could not load ${path} (${err.message}). Run something above first, or check the data source folder name.` };
  }
}

function renderAll() {
  renderOverview();
  renderDelivery();
  renderWhy();
  renderAgreement();
  mapView.refreshForDate();
}

function populateGlobalDate(preferDate) {
  const sel = el("global-date");
  const previous = sel.value;
  sel.innerHTML = "";
  if (!store.dates.length) {
    sel.innerHTML = `<option value="">No results loaded</option>`;
    store.selectedDate = "";
    return;
  }
  store.dates.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = prettyDate(d);
    sel.appendChild(opt);
  });
  // Prefer the date just computed, then whatever was already selected, then
  // the most recent date in the run.
  const pick = [preferDate, previous].find((d) => d && store.dates.includes(d)) || store.dates[store.dates.length - 1];
  sel.value = pick;
  store.selectedDate = pick;
}

async function loadAllTabs(folder, { preferDate } = {}) {
  const baseUrl = `${DATA_ROOT}/${folder}`;
  el("data-source-hint").textContent = `Loading from ${baseUrl}/ ...`;

  const [decisions, weights, crosscheck] = await Promise.all([
    fetchCsv(`${baseUrl}/consolidated_mpc_decisions.csv`),
    fetchCsv(`${baseUrl}/consolidated_topsis_weights_log.csv`),
    fetchCsv(`${baseUrl}/consolidated_module3_crosscheck.csv`),
  ]);

  store.decisions = decisions.rows;
  store.weights = weights.rows;
  store.crosscheck = crosscheck.rows;
  store.dates = [...new Set(store.decisions.map((r) => String(r.date)))].filter(Boolean).sort();

  decisionsView.setRows(decisions.rows, decisions.error);
  weightsView.setRows(weights.rows, weights.error);
  crosscheckView.setRows(crosscheck.rows, crosscheck.error);

  populateGlobalDate(preferDate);
  mapView.setReleases(store.decisions);

  el("data-source-hint").textContent = decisions.error
    ? decisions.error
    : `${store.dates.length} date${store.dates.length === 1 ? "" : "s"} loaded from ${baseUrl}/`;

  renderAll();
}

async function loadUiConfig() {
  try {
    const res = await fetch(`${API_BASE}/api/config`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg = await res.json();
    if (cfg && cfg.base_weights) uiConfig = { ...uiConfig, ...cfg };
  } catch (err) {
    // Keep the built-in defaults; they match config.py as shipped. Only the
    // "vs baseline" comparisons would be affected if the pipeline was retuned.
    console.warn("Could not load /api/config, using built-in defaults:", err.message);
  }
}

function wireDataSource() {
  el("data-source-load").addEventListener("click", () => {
    const folder = el("data-source").value.trim() || "outputs";
    loadAllTabs(folder);
  });
}

function wireGlobalDate() {
  el("global-date").addEventListener("change", (e) => {
    store.selectedDate = e.target.value;
    renderAll();
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
      if (step.dataset.target === "map") mapView.onTabShown();
    });
  });
}

// ---------------------------------------------------------------------
// Map tab - satellite view of tank capacity (blue), command area (green),
// cascade connectivity (thin lines) and the release decision for the date
// chosen in the shared date bar. Geo/sizing data comes from mpc_api.py's
// /api/tanks/geo (reuses the existing loaders server-side, no hydrology math
// duplicated here); release-by-date comes from the same `store` every other
// tab reads, so the map can never show a different date than the page header.
// ---------------------------------------------------------------------

const mapView = (() => {
  const ACRE_TO_SQM = 4046.8564224;
  const MIN_CAP_RADIUS_M = 25;
  const MAX_CAP_RADIUS_M = 220;
  // Fixed default zoom, not an auto-fit-everything zoom. fitBounds() was
  // zooming out far enough to fit all 32 tanks at once, which shrank every
  // circle/arc/number below legible size - reading the % required manually
  // zooming in every time. This trades "see everything at once" for
  // "readable by default"; pan/zoom out for the wider view when wanted.
  const DEFAULT_ZOOM = 15;

  let map = null;
  let geo = null;                 // { tanks: [...], edges: [...] }
  let maxSMax = 1;
  let releasesByDate = {};        // { "2026-07-29": { tank_id: {release, demand} } }
  let layers = { edges: [], arrows: [], commandCircles: [], capCircles: [], nameLabels: [] };
  let tanksById = {};             // tank_id -> tank record from /api/tanks/geo
  let selectedTankId = null;      // which tank the detail panel is currently showing
  let initStarted = false;

  function commandAreaRadiusM(acres) {
    const areaM2 = Math.max(0, acres) * ACRE_TO_SQM;
    return Math.sqrt(areaM2 / Math.PI);
  }

  function capacityRadiusM(sMax) {
    const t = Math.sqrt(Math.max(0, sMax)) / Math.sqrt(maxSMax || 1);
    return MIN_CAP_RADIUS_M + (MAX_CAP_RADIUS_M - MIN_CAP_RADIUS_M) * t;
  }

  function clearLayers() {
    Object.values(layers).flat().forEach((l) => map.removeLayer(l));
    layers = { edges: [], arrows: [], commandCircles: [], capCircles: [], nameLabels: [] };
  }

  function ensureMap() {
    if (map) return;
    map = L.map("map-container", { minZoom: 11 });
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri - Esri, Maxar, Earthstar Geographics",
        maxZoom: 18,
      }
    ).addTo(map);

    const legend = L.control({ position: "bottomright" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "map-legend");
      div.innerHTML = `
        <div><span class="swatch" style="background:#3b82f6"></span>Tank capacity (S_max)</div>
        <div><span class="swatch" style="background:#22c55e"></span>Command area (demand basis)</div>
        <div><span style="display:inline-block;width:14px;height:0;border-top:2px solid #fff;opacity:.6;margin-right:6px;vertical-align:middle;"></span>Cascade connectivity (arrow = flow direction)</div>
        <div>Hover a tank for a quick card; click for the full breakdown, right &rarr;</div>
      `;
      return div;
    };
    legend.addTo(map);
  }

  async function fetchGeo() {
    if (geo) return geo;
    const res = await fetch(`${API_BASE}/api/tanks/geo`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    geo = await res.json();
    maxSMax = Math.max(...geo.tanks.map((t) => t.s_max_m3), 1);
    tanksById = {};
    geo.tanks.forEach((t) => (tanksById[t.tank_id] = t));
    return geo;
  }

  function drawStatic() {
    clearLayers();

    // Connectivity lines first, so circles sit visually on top of them.
    // e.from/e.to are the true upstream->downstream direction (see
    // mpc_api.py's _tanks_geo_payload - alpha's row=destination,
    // column=source convention, matching state_transition.py's mass
    // balance). A small low-opacity arrowhead at the midpoint shows which
    // way water actually moves, without making the line itself any louder.
    const canDecorate = typeof L.polylineDecorator === "function";
    geo.edges.forEach((e) => {
      const a = tanksById[e.from], b = tanksById[e.to];
      if (!a || !b) return;
      const path = [[a.lat, a.lon], [b.lat, b.lon]];
      const line = L.polyline(path, {
        color: "#ffffff",
        weight: 1.5,
        opacity: 0.25,
        interactive: false,
      }).addTo(map);
      layers.edges.push(line);

      if (canDecorate) {
        const arrow = L.polylineDecorator(line, {
          patterns: [{
            offset: "50%",
            repeat: 0,
            symbol: L.Symbol.arrowHead({
              pixelSize: 9,
              polygon: false,
              pathOptions: { color: "#ffffff", opacity: 0.4, weight: 1.5, interactive: false },
            }),
          }],
        }).addTo(map);
        layers.arrows.push(arrow);
      }
    });

    geo.tanks.forEach((t) => {
      const commandCircle = L.circle([t.lat, t.lon], {
        radius: commandAreaRadiusM(t.command_area_acres),
        color: "#22c55e",
        weight: 1.5,
        opacity: 0.6,
        fillColor: "#22c55e",
        fillOpacity: 0.12,
      }).addTo(map);
      commandCircle.bindTooltip(() => popupHtml(t), {
        direction: "top", className: "tank-tooltip-rich", opacity: 1,
      });
      commandCircle.on("click", () => showTankDetail(t.tank_id));
      layers.commandCircles.push(commandCircle);

      const capCircle = L.circle([t.lat, t.lon], {
        radius: capacityRadiusM(t.s_max_m3),
        color: "#3b82f6",
        weight: 1.5,
        opacity: 0.85,
        fillColor: "#3b82f6",
        fillOpacity: 0.4,
      }).addTo(map);
      capCircle.bindTooltip(() => popupHtml(t), {
        direction: "top", className: "tank-tooltip-rich", opacity: 1,
      });
      capCircle.on("click", () => showTankDetail(t.tank_id));
      layers.capCircles.push(capCircle);

      // Always-visible name - without this there was no way to tell tanks
      // apart without hovering each one individually.
      const nameLabel = L.marker([t.lat, t.lon], {
        icon: L.divIcon({
          className: "tank-name-label",
          html: esc(tankLabel(t.tank_id)),
          iconSize: null,
          iconAnchor: [-6, -4],   // small offset up-right, clear of the click target at center
        }),
        interactive: false,
      }).addTo(map);
      layers.nameLabels.push(nameLabel);
    });

    const bounds = L.latLngBounds(geo.tanks.map((t) => [t.lat, t.lon]));
    map.setView(bounds.getCenter(), DEFAULT_ZOOM);
  }

  // Release-vs-demand PAIRED bar: two separate columns (demand, release) side
  // by side, each scaled to that tank's own max(demand, release) so both are
  // directly comparable regardless of tank size. Two peer bars read more
  // literally than one overlaid on the other - the shorter bar next to the
  // taller one still makes the shortfall obvious without implying one is
  // "behind" the other.
  function releaseDemandBarsHtml(entry, { big } = {}) {
    const size = big ? "big" : "";
    if (!entry || entry.release === undefined || entry.release === null) {
      return `<div class="tank-pair ${size}"><div class="tank-pair-nodata">no data</div></div>`;
    }
    const release = Number(entry.release);
    const demand = entry.demand !== null && entry.demand !== undefined ? Number(entry.demand) : null;

    if (demand === null || demand <= 0) {
      // No demand figure saved for this run - show release alone rather than
      // fabricate a demand bar that was never recorded.
      const bars = `<div class="tank-pair ${size}">
        <div class="pair-bar demand" style="height:0%"></div>
        <div class="pair-bar release" style="height:100%"></div>
      </div>`;
      return big
        ? `${bars}<div class="pair-caption">${n0(release)} m&sup3; released (demand not recorded)</div>`
        : bars;
    }

    const maxVal = Math.max(demand, release, 1);
    const demandPct = (demand / maxVal) * 100;
    const releasePct = (release / maxVal) * 100;
    const over = release > demand;
    const bars = `<div class="tank-pair ${size}">
      <div class="pair-bar demand" style="height:${demandPct}%"></div>
      <div class="pair-bar release ${over ? "over" : ""}" style="height:${releasePct}%"></div>
    </div>`;
    if (!big) return bars;

    const caption = over
      ? `Release ${n0(release)} m&sup3; exceeded demand ${n0(demand)} m&sup3;`
      : release < demand
        ? `${n0(demand - release)} m&sup3; unmet (${n0(release)} of ${n0(demand)} m&sup3;)`
        : `Demand fully met - ${n0(release)} m&sup3;`;
    return `${bars}
      <div class="pair-legend">
        <span><span class="swatch-sq demand"></span>Demand ${n0(demand)} m&sup3;</span>
        <span><span class="swatch-sq release${over ? " over" : ""}"></span>Release ${n0(release)} m&sup3;</span>
      </div>
      <div class="pair-caption">${caption}</div>`;
  }

  // % of demand met - used by the detail panel's history table (tierClass
  // below maps it to a color, validated earlier with the dataviz skill's
  // validate_palette.js: green/amber/red all pairwise pass CVD separation).
  function satisfactionPct(entry) {
    if (!entry || entry.release === undefined || entry.release === null) return null;
    const demand = entry.demand !== null && entry.demand !== undefined ? Number(entry.demand) : null;
    if (demand === null || demand <= 0) return null;
    return (Number(entry.release) / demand) * 100;
  }

  function popupHtml(t) {
    const date = store.selectedDate;
    const entry = releasesByDate[date] ? releasesByDate[date][t.tank_id] : undefined;
    return `<div class="map-popup">
      <h4>${esc(tankLabel(t.tank_id))}</h4>
      <table>
        <tr><td class="k">Capacity (S_max)</td><td class="v">${n0(t.s_max_m3)} m&sup3;</td></tr>
        <tr><td class="k">Command area</td><td class="v">${t.command_area_acres.toFixed(1)} acres</td></tr>
        <tr><td class="k">Catchment area</td><td class="v">${t.catchment_area_km2.toFixed(2)} km&sup2;</td></tr>
      </table>
      <p class="map-popup-label">Release vs. demand${date ? " on " + prettyDate(date) : ""}</p>
      ${releaseDemandBarsHtml(entry, { big: true })}
    </div>`;
  }

  function tierClass(pct) {
    if (pct === null) return "";
    if (pct >= 90) return "tier-good";
    if (pct >= 50) return "tier-warn";
    return "tier-bad";
  }

  // Full breakdown for one tank, opened in the right-hand panel on click.
  // Reuses popupHtml's stats + the big paired-bar for the selected date, and
  // adds a history table across every date currently loaded (releasesByDate
  // already holds all of them in memory - no extra fetch needed).
  function showTankDetail(tankId) {
    selectedTankId = tankId;
    renderDetailPanel();
  }

  function renderDetailPanel() {
    const panel = el("tank-detail-panel");
    const isOpen = !!(selectedTankId && tanksById[selectedTankId]);

    // The panel takes zero layout space until a tank is picked - the map
    // stays full-width until then. Toggling it resizes #map-container, so
    // Leaflet needs an explicit invalidateSize() once the CSS reflow settles,
    // or it keeps rendering tiles for the old size/position.
    el("map-row").classList.toggle("panel-open", isOpen);
    if (map) setTimeout(() => map.invalidateSize(), 50);

    if (!isOpen) {
      panel.innerHTML = `<div class="detail-placeholder">Click a tank on the map to see its full demand/release breakdown here.</div>`;
      return;
    }
    const t = tanksById[selectedTankId];
    const date = store.selectedDate;
    const entry = releasesByDate[date] ? releasesByDate[date][t.tank_id] : undefined;

    const dates = Object.keys(releasesByDate).sort().reverse();
    const historyRows = dates.map((d) => {
      const e = releasesByDate[d][t.tank_id];
      if (!e || e.release === undefined || e.release === null) {
        return `<tr><td>${d}</td><td colspan="3">no data</td></tr>`;
      }
      const pct = satisfactionPct(e);
      const pctText = pct === null ? "-" : `${Math.round(pct)}%`;
      return `<tr class="${d === date ? "current-date" : ""}">
        <td>${d}</td>
        <td>${n0(e.demand ?? 0)}</td>
        <td>${n0(e.release)}</td>
        <td class="${tierClass(pct)}">${pctText}</td>
      </tr>`;
    }).join("");

    panel.innerHTML = `<div class="map-popup detail-body">
      <div class="detail-header">
        <h4>${esc(tankLabel(t.tank_id))}</h4>
        <button type="button" class="detail-close" title="Close">&times;</button>
      </div>
      <table>
        <tr><td class="k">Capacity (S_max)</td><td class="v">${n0(t.s_max_m3)} m&sup3;</td></tr>
        <tr><td class="k">Command area</td><td class="v">${t.command_area_acres.toFixed(1)} acres</td></tr>
        <tr><td class="k">Catchment area</td><td class="v">${t.catchment_area_km2.toFixed(2)} km&sup2;</td></tr>
      </table>
      <p class="map-popup-label">Release vs. demand${date ? " on " + prettyDate(date) : ""}</p>
      ${releaseDemandBarsHtml(entry, { big: true })}
      ${dates.length ? `
        <p class="map-popup-label">History (${dates.length} date${dates.length > 1 ? "s" : ""} loaded)</p>
        <table class="detail-history">
          <thead><tr><th>Date</th><th>Demand</th><th>Release</th><th>% met</th></tr></thead>
          <tbody>${historyRows}</tbody>
        </table>
      ` : ""}
    </div>`;

    panel.querySelector(".detail-close").addEventListener("click", () => {
      selectedTankId = null;
      renderDetailPanel();
    });
  }

  /** Rebuild the by-date lookup from the rows loadAllTabs already parsed. */
  function setReleases(rows) {
    releasesByDate = {};
    rows.forEach((row) => {
      const d = String(row.date);
      if (!releasesByDate[d]) releasesByDate[d] = {};
      releasesByDate[d][row.tank_id] = {
        release: row.release_m3,
        demand: row.demand_m3 !== undefined && row.demand_m3 !== null ? row.demand_m3 : null,
      };
    });
    const count = Object.keys(releasesByDate).length;
    el("map-hint").textContent = count
      ? `${count} date(s) available`
      : "No release data yet - run a job above. The map still shows tank and command-area sizing.";
    if (map) renderDetailPanel();
  }

  async function init() {
    if (initStarted) return;
    initStarted = true;
    try {
      if (typeof L === "undefined") {
        throw new Error("Leaflet failed to load from the CDN (unpkg.com) - check your network/firewall, or that you have internet access, then click 'Reload map data'");
      }
      ensureMap();
      await fetchGeo();
      drawStatic();
    } catch (err) {
      initStarted = false;   // let "Reload map data" retry instead of being permanently stuck
      el("map-hint").textContent = `Map failed to load: ${err.message}`;
      console.error("mapView.init failed:", err);
    }
  }

  function onTabShown() {
    init().then(() => {
      if (map) setTimeout(() => map.invalidateSize(), 50);
    });
  }

  return {
    onTabShown,
    setReleases,
    reload: () => { geo = null; initStarted = false; init(); },
    refreshForDate: () => { if (map && geo) renderDetailPanel(); },
    // Tank sizing (s_max_m3) is useful outside the map too (see
    // largeReservoirIds() / the Overview "village tanks" comparison) -
    // exposed here rather than fetched a second time, since fetchGeo()
    // already caches it and has no Leaflet dependency of its own.
    ensureGeo: fetchGeo,
    getTanksById: () => tanksById,
  };
})();

function wireMapControls() {
  el("map-reload-btn").addEventListener("click", () => mapView.reload());
}

wireDailyRun();
wireSeasonRun();
wireStopButton("daily");
wireStopButton("season");
wireDataSource();
wireGlobalDate();
wireTabs();
wireMapControls();

// Config first so the "vs baseline" comparisons in Why-this-plan are right on
// the very first render, then the run data.
loadUiConfig().then(() => loadAllTabs("outputs"));

// Tank capacity (for the Overview "village tanks" comparison) fetched
// eagerly rather than waiting for the Map tab - runs in parallel with the
// run data above, so whichever finishes second re-renders Overview to pick
// up whatever the other one was missing.
mapView.ensureGeo()
  .then(() => { if (store.dates.length) renderOverview(); })
  .catch((err) => console.warn("Could not load tank sizing for the village-tanks comparison:", err.message));
