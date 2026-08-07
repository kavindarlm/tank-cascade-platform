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
  mapView.loadReleases(baseUrl);
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
      if (step.dataset.target === "map") mapView.onTabShown();
    });
  });
}

// ---------------------------------------------------------------------
// Map tab - satellite view of tank capacity (blue), command area (green),
// cascade connectivity (thin lines) and the release decision for a chosen
// date. Geo/sizing data comes from mpc_api.py's /api/tanks/geo (reuses the
// existing loaders server-side, no hydrology math duplicated here);
// release-by-date comes from whatever consolidated_mpc_decisions.csv the
// "Data source folder" box above already points at.
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
          html: t.tank_id.replace(/_/g, " "),
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
        ? `${bars}<div class="pair-caption">${Math.round(release).toLocaleString()} m&sup3; released (demand not recorded)</div>`
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
      ? `Release ${Math.round(release).toLocaleString()} m&sup3; exceeded demand ${Math.round(demand).toLocaleString()} m&sup3;`
      : release < demand
        ? `${Math.round(demand - release).toLocaleString()} m&sup3; unmet (${Math.round(release).toLocaleString()} of ${Math.round(demand).toLocaleString()} m&sup3;)`
        : `Demand fully met - ${Math.round(release).toLocaleString()} m&sup3;`;
    return `${bars}
      <div class="pair-legend">
        <span><span class="swatch-sq demand"></span>Demand ${Math.round(demand).toLocaleString()} m&sup3;</span>
        <span><span class="swatch-sq release${over ? " over" : ""}"></span>Release ${Math.round(release).toLocaleString()} m&sup3;</span>
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
    const date = el("map-date").value;
    const entry = releasesByDate[date] ? releasesByDate[date][t.tank_id] : undefined;
    return `<div class="map-popup">
      <h4>${t.tank_id}</h4>
      <table>
        <tr><td class="k">Capacity (S_max)</td><td class="v">${Math.round(t.s_max_m3).toLocaleString()} m&sup3;</td></tr>
        <tr><td class="k">Command area</td><td class="v">${t.command_area_acres.toFixed(1)} acres</td></tr>
        <tr><td class="k">Catchment area</td><td class="v">${t.catchment_area_km2.toFixed(2)} km&sup2;</td></tr>
      </table>
      <p class="map-popup-label">Release vs. demand${date ? " on " + date : ""}</p>
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
    const date = el("map-date").value;
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
        <td>${Math.round(e.demand ?? 0).toLocaleString()}</td>
        <td>${Math.round(e.release).toLocaleString()}</td>
        <td class="${tierClass(pct)}">${pctText}</td>
      </tr>`;
    }).join("");

    panel.innerHTML = `<div class="map-popup detail-body">
      <div class="detail-header">
        <h4>${t.tank_id.replace(/_/g, " ")}</h4>
        <button type="button" class="detail-close" title="Close">&times;</button>
      </div>
      <table>
        <tr><td class="k">Capacity (S_max)</td><td class="v">${Math.round(t.s_max_m3).toLocaleString()} m&sup3;</td></tr>
        <tr><td class="k">Command area</td><td class="v">${t.command_area_acres.toFixed(1)} acres</td></tr>
        <tr><td class="k">Catchment area</td><td class="v">${t.catchment_area_km2.toFixed(2)} km&sup2;</td></tr>
      </table>
      <p class="map-popup-label">Release vs. demand${date ? " on " + date : ""}</p>
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

  function populateDateSelect() {
    const sel = el("map-date");
    const dates = Object.keys(releasesByDate).sort();
    const current = sel.value;
    sel.innerHTML = "";
    if (!dates.length) {
      sel.innerHTML = `<option value="">No release data loaded</option>`;
      return;
    }
    dates.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d;
      opt.textContent = d;
      sel.appendChild(opt);
    });
    sel.value = dates.includes(current) ? current : dates[dates.length - 1];
  }

  async function loadReleases(baseUrl) {
    el("map-hint").textContent = "Loading release data...";
    try {
      const res = await fetch(`${baseUrl}/consolidated_mpc_decisions.csv`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
      releasesByDate = {};
      parsed.data.forEach((row) => {
        const d = String(row.date);
        if (!releasesByDate[d]) releasesByDate[d] = {};
        releasesByDate[d][row.tank_id] = {
          release: row.release_m3,
          demand: row.demand_m3 !== undefined && row.demand_m3 !== null ? row.demand_m3 : null,
        };
      });
      populateDateSelect();
      el("map-hint").textContent = `${Object.keys(releasesByDate).length} date(s) available`;
      if (map) renderDetailPanel();
    } catch (err) {
      releasesByDate = {};
      populateDateSelect();
      el("map-hint").textContent = `No release data yet (${err.message}) - run a job above, or the map still shows tank/command-area sizing without release labels.`;
    }
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
    loadReleases,
    reload: () => { geo = null; initStarted = false; init(); },
    refreshForDate: () => { if (map && geo) renderDetailPanel(); },
  };
})();

function wireMapControls() {
  el("map-date").addEventListener("change", () => mapView.refreshForDate());
  el("map-reload-btn").addEventListener("click", () => mapView.reload());
}

wireDailyRun();
wireSeasonRun();
wireStopButton("daily");
wireStopButton("season");
wireDataSource();
wireTabs();
wireMapControls();
loadAllTabs("outputs");
