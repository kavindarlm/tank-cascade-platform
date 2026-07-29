// Reads the Module 2 cascade edge table directly from disk -- no backend needed.
const CSV_PATH = "../Module2/development-history/outputs/alpha_edges.csv";

const VIZ_PATHS = {
  flowmap: "../Module2/development-history/outputs/cascade_geographic_flow_map.html",
  graph: "../Module2/development-history/outputs/cascade_influence_map_final.html",
};

const VIEW_META = {
  table: {
    title: "Cascade edge table",
    description: "Browse the tank-to-tank connections, filter the records, and export the current subset.",
  },
  flowmap: {
    title: "Tank system flow map",
    description: "See the physical conveyance routes as a geographic flow diagram with the same tank network.",
  },
  graph: {
    title: "Tank connectivity graph",
    description: "Explore the combined influence and conveyance graph for ranking, stability, and transfer strength.",
  },
};

let allRows = [];
let filteredRows = [];
let currentPage = 1;
let pageSize = 25;

const el = (id) => document.getElementById(id);

function initSteps() {
  const steps = document.querySelectorAll(".step-item");
  const loadedFrames = new Set();
  const pageTitleEl = el("page-title");
  const pageDescEl = el("page-description");
  const titleEl = el("active-view-title");
  const descEl = el("active-view-description");

  function activate(step) {
    const target = step.dataset.target;
    const meta = VIEW_META[target];

    steps.forEach((s) => s.classList.toggle("active", s === step));
    document.querySelectorAll(".view-panel").forEach((panel) => {
      panel.hidden = panel.id !== `panel-${target}`;
    });

    if (pageTitleEl && meta) pageTitleEl.textContent = meta.title;
    if (pageDescEl && meta) pageDescEl.textContent = meta.description;
    if (titleEl && meta) titleEl.textContent = meta.title;
    if (descEl && meta) descEl.textContent = meta.description;

    if (VIZ_PATHS[target] && !loadedFrames.has(target)) {
      el(`frame-${target}`).src = VIZ_PATHS[target];
      loadedFrames.add(target);
    }

    document.title = `${meta ? meta.title : step.dataset.title || "Tank connectivity"} - Tank Cascade Platform`;
  }

  steps.forEach((step) => step.addEventListener("click", () => activate(step)));

  const initialStep = document.querySelector(".step-item.active") || steps[0];
  if (initialStep) {
    activate(initialStep);
  }
}

initSteps();

async function loadData() {
  try {
    const res = await fetch(CSV_PATH);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
    allRows = parsed.data;
    populateFilterOptions();
    applyFilters();
  } catch (err) {
    el("table-info").textContent =
      `Could not load ${CSV_PATH} (${err.message}). Make sure you're viewing this ` +
      `through a local server, not by double-clicking the HTML file -- see the README.`;
    el("table-body").innerHTML = `<tr class="empty-row"><td colspan="6">No data loaded</td></tr>`;
  }
}

function uniqueSorted(rows, field) {
  return [...new Set(rows.map((r) => r[field]).filter((v) => v !== null && v !== undefined && v !== ""))].sort();
}

function populateFilterOptions() {
  fillSelect("filter-from-tank", uniqueSorted(allRows, "from_tank"), "All tanks");
  fillSelect("filter-to-tank", uniqueSorted(allRows, "to_tank"), "All tanks");
}

function fillSelect(id, values, placeholder) {
  const sel = el(id);
  sel.innerHTML = `<option value="">${placeholder}</option>`;
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
}

function applyFilters() {
  const fromTank = el("filter-from-tank").value;
  const toTank = el("filter-to-tank").value;

  filteredRows = allRows.filter((r) => {
    if (fromTank && r.from_tank !== fromTank) return false;
    if (toTank && r.to_tank !== toTank) return false;
    return true;
  });

  currentPage = 1;
  renderTable();
}

function fmt(v, digits) {
  if (v === null || v === undefined || v === "") return "";
  if (digits !== undefined && typeof v === "number") return v.toFixed(digits);
  return v;
}

function renderTable() {
  const total = filteredRows.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  currentPage = Math.min(currentPage, totalPages);

  const start = (currentPage - 1) * pageSize;
  const pageRows = filteredRows.slice(start, start + pageSize);

  const body = el("table-body");
  if (pageRows.length === 0) {
    body.innerHTML = `<tr class="empty-row"><td colspan="6">No rows match the current filters</td></tr>`;
  } else {
    body.innerHTML = pageRows
      .map((r) => {
        return `<tr>
          <td>${fmt(r.from_tank)}</td>
          <td>${fmt(r.to_tank)}</td>
          <td>${fmt(r.stream_path_length_m, 1)}</td>
          <td>${fmt(r["slope_m/m"], 6)}</td>
          <td>${fmt(r.effective_length_m, 2)}</td>
          <td>${fmt(r.alpha_water_transfer_coefficient, 4)}</td>
        </tr>`;
      })
      .join("");
  }

  el("table-info").textContent =
    total === 0 ? "0 rows" : `Showing ${start + 1}-${Math.min(start + pageSize, total)} of ${total} rows`;
  el("page-indicator").textContent = `Page ${currentPage} of ${totalPages}`;
  el("btn-prev").disabled = currentPage <= 1;
  el("btn-next").disabled = currentPage >= totalPages;
}

function downloadCSV() {
  const csv = Papa.unparse(filteredRows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `alpha_edges_filtered_${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

el("filter-from-tank").addEventListener("change", applyFilters);
el("filter-to-tank").addEventListener("change", applyFilters);

el("btn-clear").addEventListener("click", () => {
  el("filter-from-tank").value = "";
  el("filter-to-tank").value = "";
  applyFilters();
});

el("btn-download").addEventListener("click", downloadCSV);

el("btn-prev").addEventListener("click", () => {
  if (currentPage > 1) { currentPage--; renderTable(); }
});
el("btn-next").addEventListener("click", () => {
  currentPage++; renderTable();
});
el("page-size").addEventListener("change", (e) => {
  pageSize = parseInt(e.target.value, 10);
  currentPage = 1;
  renderTable();
});

loadData();
