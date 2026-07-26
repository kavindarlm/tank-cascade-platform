// Reads the shared Module 1 table directly from disk -- no backend needed.
// The table lives one folder up from this frontend, at ../data/tank_storage.csv
const CSV_PATH = "../Module1/live-update-service/data/tank_storage.csv";

let allRows = [];
let filteredRows = [];
let currentPage = 1;
let pageSize = 50;

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
      .map((r) => {
        const statusClass = r.record_status === "live" ? "status-live" : "status-validated";
        return `<tr>
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

loadData();
