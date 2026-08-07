// Reads the shared Module 1 table directly from disk -- no backend needed.
// The table lives one folder up from this frontend, at ../data/tank_storage.csv
const CSV_PATH = "../Module1/live-update-service/data/tank_storage.csv";

// Real digitized tank boundary polygons (same source data the GEE asset
// nachchaduwa-32-final was built from - see
// Module1/live-update-service/scripts/export_tank_boundaries.py), keyed by
// tank_id in each feature's properties.
const BOUNDARIES_PATH = "../Module1/live-update-service/data/tank_boundaries.geojson";

let allRows = [];
let filteredRows = [];
let currentPage = 1;
let pageSize = 50;

let boundariesByTankId = null; // Map<string, GeoJSON Feature>, loaded lazily
let boundariesLoadError = null;
let drawerMap = null;
let drawerLayer = null;
let drawerLabelMarker = null;
let selectedTankId = null;

const el = (id) => document.getElementById(id);

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
    el("table-body").innerHTML = `<tr class="empty-row"><td colspan="13">No data loaded</td></tr>`;
  }
}

function uniqueSorted(rows, field) {
  return [...new Set(rows.map((r) => r[field]).filter((v) => v !== null && v !== undefined && v !== ""))].sort();
}

function populateFilterOptions() {
  fillSelect("filter-tank", uniqueSorted(allRows, "pond_name"), "All tanks");
  fillSelect("filter-sensor", uniqueSorted(allRows, "sensor_used"), "All sensors");
  fillSelect("filter-status", uniqueSorted(allRows, "record_status"), "All");
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
  const tank = el("filter-tank").value;
  const from = el("filter-from").value;
  const to = el("filter-to").value;
  const sensor = el("filter-sensor").value;
  const status = el("filter-status").value;

  filteredRows = allRows.filter((r) => {
    if (tank && r.pond_name !== tank) return false;
    if (sensor && r.sensor_used !== sensor) return false;
    if (status && r.record_status !== status) return false;
    if (from && String(r.date) < from) return false;
    if (to && String(r.date) > to) return false;
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
    body.innerHTML = `<tr class="empty-row"><td colspan="13">No rows match the current filters</td></tr>`;
  } else {
    body.innerHTML = pageRows
      .map((r, i) => {
        const statusClass = r.record_status === "live" ? "status-live" : "status-validated";
        const selectedClass = String(r.tank_id) === String(selectedTankId) ? " row-selected" : "";
        return `<tr data-row-index="${start + i}" class="${selectedClass.trim()}">
          <td>${fmt(r.tank_id)}</td>
          <td>${fmt(r.pond_name)}</td>
          <td>${fmt(r.date)}</td>
          <td>${fmt(r.water_ha, 2)}</td>
          <td>${fmt(r.total_ha, 2)}</td>
          <td>${fmt(r.fill_pct, 1)}</td>
          <td>${fmt(r.sensor_used)}</td>
          <td>${fmt(r.s2_date)}</td>
          <td>${fmt(r.day_gap)}</td>
          <td>${fmt(r.volume_mcm, 4)}</td>
          <td>${fmt(r.volume_confidence)}</td>
          <td class="${statusClass}">${fmt(r.record_status)}</td>
          <td>${fmt(r.last_updated)}</td>
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
  a.download = `tank_storage_filtered_${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

el("filter-tank").addEventListener("change", applyFilters);
el("filter-from").addEventListener("change", applyFilters);
el("filter-to").addEventListener("change", applyFilters);
el("filter-sensor").addEventListener("change", applyFilters);
el("filter-status").addEventListener("change", applyFilters);

el("btn-clear").addEventListener("click", () => {
  el("filter-tank").value = "";
  el("filter-from").value = "";
  el("filter-to").value = "";
  el("filter-sensor").value = "";
  el("filter-status").value = "";
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

// ---------------------------------------------------------------------
// Row -> tank detail drawer (map + details), same interaction pattern as
// Module 4's map tab (frontend/mpc.js) - click to open a detail view,
// close via the X, backdrop click, or Escape.
// ---------------------------------------------------------------------

async function loadBoundaries() {
  if (boundariesByTankId) return boundariesByTankId;
  try {
    const res = await fetch(BOUNDARIES_PATH);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();
    boundariesByTankId = new Map(
      geojson.features.map((f) => [String(f.properties.tank_id), f])
    );
  } catch (err) {
    boundariesLoadError = err.message;
    boundariesByTankId = new Map();
  }
  return boundariesByTankId;
}

function ensureMap() {
  if (drawerMap) return drawerMap;
  drawerMap = L.map("tank-drawer-map", { attributionControl: false, zoomControl: true });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
  }).addTo(drawerMap);
  return drawerMap;
}

function detailItem(k, v) {
  if (v === "" || v === null || v === undefined) return "";
  return `<div class="item"><span class="k">${k}</span><span class="v">${v}</span></div>`;
}

async function renderDrawerContent(row) {
  el("tank-drawer-title").textContent = fmt(row.pond_name) || `Tank ${fmt(row.tank_id)}`;
  el("tank-drawer-subtitle").textContent = `Tank ID ${fmt(row.tank_id)} - record for ${fmt(row.date)}`;

  const statusLabel = row.record_status === "live" ? "Live (unvalidated)" : fmt(row.record_status);
  el("tank-drawer-details").innerHTML = [
    detailItem("Date", fmt(row.date)),
    detailItem("Status", statusLabel),
    detailItem("Water area", row.water_ha != null ? `${fmt(row.water_ha, 2)} ha` : ""),
    detailItem("Total tank area", row.total_ha != null ? `${fmt(row.total_ha, 2)} ha` : ""),
    detailItem("Fill %", row.fill_pct != null ? `${fmt(row.fill_pct, 1)}%` : ""),
    detailItem("Volume", row.volume_mcm != null ? `${fmt(row.volume_mcm, 4)} MCM` : ""),
    detailItem("Confidence", fmt(row.volume_confidence)),
    detailItem("Sensor used", fmt(row.sensor_used)),
    detailItem("S2 date", fmt(row.s2_date)),
    detailItem("Day gap", fmt(row.day_gap)),
    detailItem("Last updated", fmt(row.last_updated)),
  ].join("");

  const map = ensureMap();
  if (drawerLayer) {
    map.removeLayer(drawerLayer);
    drawerLayer = null;
  }
  if (drawerLabelMarker) {
    map.removeLayer(drawerLabelMarker);
    drawerLabelMarker = null;
  }

  const boundaries = await loadBoundaries();
  const feature = boundaries.get(String(row.tank_id));
  const hint = el("tank-drawer-hint");

  if (!feature) {
    hint.textContent = boundariesLoadError
      ? `Boundary data unavailable (${boundariesLoadError}).`
      : "No digitized boundary found for this tank.";
    map.setView([8.12, 80.56], 12); // Anuradhapura cascade area, generic fallback
    setTimeout(() => map.invalidateSize(), 250);
    return;
  }

  hint.textContent = "";
  drawerLayer = L.geoJSON(feature, {
    style: { color: "#2b6cb0", weight: 2, fillColor: "#4f8bc9", fillOpacity: 0.35 },
  }).addTo(map);

  const [cx, cy] = feature.properties.centroid;
  drawerLabelMarker = L.marker([cy, cx], {
    icon: L.divIcon({
      className: "tank-name-label",
      html: fmt(row.pond_name),
      iconSize: null,
      iconAnchor: [-6, -4],
    }),
    interactive: false,
  }).addTo(map);

  map.fitBounds(drawerLayer.getBounds(), { padding: [24, 24] });
  setTimeout(() => map.invalidateSize(), 250);
}

function openDrawer(row) {
  selectedTankId = row.tank_id;
  el("tank-drawer").classList.add("open");
  el("tank-drawer").setAttribute("aria-hidden", "false");
  el("tank-drawer-backdrop").classList.add("open");
  renderTable(); // refresh row highlight
  renderDrawerContent(row);
}

function closeDrawer() {
  selectedTankId = null;
  el("tank-drawer").classList.remove("open");
  el("tank-drawer").setAttribute("aria-hidden", "true");
  el("tank-drawer-backdrop").classList.remove("open");
  renderTable(); // clear row highlight
}

el("table-body").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-row-index]");
  if (!tr) return;
  const row = filteredRows[Number(tr.dataset.rowIndex)];
  if (!row) return;

  if (String(row.tank_id) === String(selectedTankId) && el("tank-drawer").classList.contains("open")) {
    closeDrawer();
    return;
  }
  openDrawer(row);
});

el("tank-drawer-close").addEventListener("click", closeDrawer);
el("tank-drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && el("tank-drawer").classList.contains("open")) closeDrawer();
});

// ---------------------------------------------------------------------
// "Tank map view" - a card grid (one card per tank, a real non-interactive
// mini map with its boundary drawn on top), swapped in for the table.
// Clicking a card opens the same detail drawer as clicking a table row.
// ---------------------------------------------------------------------

let cardMaps = []; // Leaflet map instances for the currently-built card grid

function latestRowByTankId() {
  const latest = new Map();
  for (const r of allRows) {
    const key = String(r.tank_id);
    const prev = latest.get(key);
    if (!prev || String(r.date) > String(prev.date)) latest.set(key, r);
  }
  return latest;
}

function initCardThumbMap(thumbId, feature) {
  const map = L.map(thumbId, {
    attributionControl: false,
    zoomControl: false,
    dragging: false,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    touchZoom: false,
    tap: false,
  });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);

  if (feature) {
    const layer = L.geoJSON(feature, {
      style: { color: "#2b6cb0", weight: 1.5, fillColor: "#4f8bc9", fillOpacity: 0.5 },
    }).addTo(map);
    map.fitBounds(layer.getBounds(), { padding: [10, 10] });
  } else {
    map.setView([8.12, 80.56], 12); // Anuradhapura cascade area, generic fallback
  }
  cardMaps.push(map);
  setTimeout(() => map.invalidateSize(), 60);
}

async function buildTankCards() {
  const grid = el("tank-cards-grid");
  const hint = el("tank-cards-hint");

  cardMaps.forEach((m) => m.remove());
  cardMaps = [];
  grid.innerHTML = "";
  hint.textContent = "Loading tank boundaries...";

  const boundaries = await loadBoundaries();
  const latestByTank = latestRowByTankId();

  // Card list driven by whichever tanks actually have data (tank_meta's 32
  // via the storage table), each paired with its boundary if one matched.
  const tankIds = [...latestByTank.keys()].sort((a, b) => Number(a) - Number(b));

  if (tankIds.length === 0) {
    hint.textContent = "No tank data loaded yet.";
    return;
  }

  hint.textContent = boundariesLoadError
    ? `Boundary shapes unavailable (${boundariesLoadError}) - showing map previews without an outline.`
    : "Click a tank to see its full boundary map and latest reading.";

  grid.innerHTML = tankIds
    .map((tankId) => {
      const row = latestByTank.get(tankId);
      return `<div class="tank-card" data-tank-id="${tankId}">
        <div class="tank-card-thumb" id="tank-thumb-${tankId}"></div>
        <div class="tank-card-body">
          <div class="tank-card-name">${fmt(row.pond_name) || `Tank ${tankId}`}</div>
          <div class="tank-card-stats">
            <span>Fill: <b>${row.fill_pct != null ? fmt(row.fill_pct, 1) + "%" : "-"}</b></span>
            <span>Area: <b>${row.total_ha != null ? fmt(row.total_ha, 1) + " ha" : "-"}</b></span>
            <span>As of <b>${fmt(row.date) || "-"}</b></span>
          </div>
        </div>
      </div>`;
    })
    .join("");

  // Leaflet needs the container in the DOM before it can size itself, so
  // maps are created after innerHTML is set, not inside the map() above.
  for (const tankId of tankIds) {
    initCardThumbMap(`tank-thumb-${tankId}`, boundaries.get(tankId));
  }
}

function showCardsView() {
  el("table-view").hidden = true;
  el("tank-cards-view").hidden = false;
  buildTankCards().catch((err) => {
    el("tank-cards-hint").textContent = `Could not build tank map view (${err.message}).`;
  });
}

function showTableView() {
  el("tank-cards-view").hidden = true;
  el("table-view").hidden = false;
}

el("btn-map-view").addEventListener("click", showCardsView);
el("btn-back-to-table").addEventListener("click", showTableView);

loadData();
