const state = {
  datasets: [],
  checkpoints: [],
  defaultVehicles: {},
  defaultRefPoints: {},
  selectedFilename: null,
  summary: null,
  instance: null,
  plottedPoints: [],
  eamMode: 'train',
  jobs: [],
  activeEamJobId: null,
  activeCmmaJobId: null,
  streams: {},
  resultsList: [],
  resultSummaryA: null,
  compareData: null,
  selectedSolIdx: 0,
  paretoPlottedPoints: [],
  routeData: null,
  highlightedVehicleId: null,
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatTime(ts) {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleTimeString('vi-VN');
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data;
}

function setupTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.remove('active'));
      panels.forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ==================== STEP 1 & 2: SYSTEM & DATASETS ====================

async function loadSystemStatus() {
  const grid = document.getElementById('system-status-grid');
  try {
    const s = await fetchJSON('/api/system/status');
    state.defaultVehicles = s.default_vehicles || {};
    state.defaultRefPoints = s.default_ref_points || {};

    const items = [
      { label: 'Python', value: s.python_version, sub: s.os },
      { label: 'PyTorch', value: s.torch_version, sub: s.cuda_available ? `CUDA (${s.gpu_count} GPU)` : 'CPU mode' },
      { label: 'Thiết bị tính toán', value: s.gpu_name, sub: s.cuda_available ? 'Sẵn sàng tăng tốc GPU' : 'Chạy CPU' },
      { label: 'Thư mục datasets/', value: `${s.directories.datasets.count} file .npy`, sub: s.directories.datasets.exists ? 'Đã sẵn sàng' : 'Chưa tạo' },
      { label: 'saved_model/ (E-AM)', value: `${s.directories.saved_model.checkpoints} checkpoint`, sub: s.directories.saved_model.exists ? 'Đã tồn tại' : 'Chưa có model' },
      { label: 'result/ (CMMA)', value: `${s.directories.result.runs} lần chạy`, sub: s.directories.result.exists ? 'Đã tồn tại' : 'Chưa chạy' },
    ];
    grid.innerHTML = items
      .map(
        (item) => `
        <div class="metric">
          <div class="metric-label">${item.label}</div>
          <div class="metric-value">${item.value}</div>
          <div class="metric-sub">${item.sub}</div>
        </div>`
      )
      .join('');

    const capSelect = document.getElementById('gen-customer-size');
    if (capSelect.options.length === 0) {
      const sizes = Object.keys(s.capacities).map(Number).sort((a, b) => a - b);
      capSelect.innerHTML = sizes
        .map((sz) => `<option value="${sz}" ${sz === 20 ? 'selected' : ''}>${sz} khách (cap=${s.capacities[sz]})</option>`)
        .join('');
    }
  } catch (err) {
    grid.innerHTML = `<div class="alert alert-error">Lỗi tải trạng thái hệ thống: ${err.message}</div>`;
  }
}

async function loadDatasets(preferredName = null) {
  const data = await fetchJSON('/api/datasets');
  state.datasets = data.datasets;

  const tbody = document.querySelector('#datasets-table tbody');
  const select = document.getElementById('select-dataset');

  populateRunnerDatasetSelects();

  if (state.datasets.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted">Chưa có file .npy nào trong datasets/</td></tr>';
    select.innerHTML = '';
    return;
  }

  if (preferredName && state.datasets.some((d) => d.filename === preferredName)) {
    state.selectedFilename = preferredName;
  } else if (!state.selectedFilename || !state.datasets.some((d) => d.filename === state.selectedFilename)) {
    const def = state.datasets.find((d) => d.filename === 'cvrp20-v4.npy') || state.datasets[0];
    state.selectedFilename = def.filename;
  }

  select.innerHTML = state.datasets
    .map((d) => `<option value="${d.filename}" ${d.filename === state.selectedFilename ? 'selected' : ''}>${d.filename} (${d.num_instances} bài, ${d.customer_size} khách)</option>`)
    .join('');

  renderDatasetsTable();
  await selectDataset(state.selectedFilename);
}

function renderDatasetsTable() {
  const tbody = document.querySelector('#datasets-table tbody');
  tbody.innerHTML = state.datasets
    .map(
      (d) => `
      <tr data-file="${d.filename}" class="${d.filename === state.selectedFilename ? 'selected' : ''}">
        <td><strong>${d.filename}</strong></td>
        <td>${d.kind}</td>
        <td>${d.num_instances}</td>
        <td>${d.customer_size}</td>
        <td>${d.vehicle_num ?? 'Tự nhập'}</td>
        <td>${d.auto_zoom ? 'Có' : 'Không'}</td>
        <td><code>[${d.shape.join(', ')}]</code></td>
        <td>${formatBytes(d.size_bytes)}</td>
      </tr>`
    )
    .join('');

  tbody.querySelectorAll('tr').forEach((row) => {
    row.addEventListener('click', () => {
      const file = row.dataset.file;
      document.getElementById('select-dataset').value = file;
      selectDataset(file);
    });
  });
}

async function selectDataset(filename) {
  state.selectedFilename = filename;
  renderDatasetsTable();

  const summary = await fetchJSON(`/api/datasets/${encodeURIComponent(filename)}/summary`);
  state.summary = summary;
  const meta = summary.meta;

  const slider = document.getElementById('instance-slider');
  const numInput = document.getElementById('instance-index');
  const maxIdx = Math.max(0, meta.num_instances - 1);
  slider.max = maxIdx;
  numInput.max = maxIdx;
  slider.value = 0;
  numInput.value = 0;
  slider.disabled = maxIdx === 0;
  numInput.disabled = maxIdx === 0;

  document.getElementById('chk-zoom').checked = meta.auto_zoom || summary.out_of_unit_box;
  document.getElementById('chk-ids').checked = meta.customer_size <= 50;

  const metricsEl = document.getElementById('dataset-summary-metrics');
  metricsEl.innerHTML = `
    <div class="metric"><div class="metric-label">Số bài toán</div><div class="metric-value">${meta.num_instances}</div></div>
    <div class="metric"><div class="metric-label">Số khách hàng</div><div class="metric-value">${meta.customer_size}</div></div>
    <div class="metric"><div class="metric-label">Số xe gợi ý</div><div class="metric-value">${meta.vehicle_num ?? 'N/A'}</div></div>
    <div class="metric"><div class="metric-label">Phạm vi X / Y gốc</div><div class="metric-value" style="font-size:13px">[${summary.x_min.toFixed(2)}, ${summary.x_max.toFixed(2)}] × [${summary.y_min.toFixed(2)}, ${summary.y_max.toFixed(2)}]</div></div>
    ${summary.has_demand ? `<div class="metric"><div class="metric-label">Tổng demand TB / bài</div><div class="metric-value">${summary.total_demand_mean.toFixed(2)}</div><div class="metric-sub">Min ${summary.total_demand_min.toFixed(2)} — Max ${summary.total_demand_max.toFixed(2)}</div></div>` : ''}
  `;

  const alertsEl = document.getElementById('dataset-alerts');
  const alerts = [];
  if (summary.out_of_unit_box) {
    alerts.push('<div class="alert alert-info">Tọa độ gốc nằm ngoài [0, 1]². Đã tự động bật chế độ Zoom để chuẩn hóa khi xem trước (khớp với <code>--zoom auto</code> trong <code>src/options.py</code>).</div>');
  }
  if (summary.has_demand && !summary.depot_demand_ok) {
    alerts.push('<div class="alert alert-warn">Cảnh báo: phát hiện Node 0 (Depot) có demand khác 0.</div>');
  }
  if (summary.feasible_count !== null && summary.feasible_count !== undefined && summary.feasible_count < meta.num_instances) {
    alerts.push(`<div class="alert alert-warn">Có ${meta.num_instances - summary.feasible_count}/${meta.num_instances} bài toán có tổng demand vượt ngưỡng <code>vehicle_num - 1</code> với ${meta.vehicle_num} xe.</div>`);
  }
  alertsEl.innerHTML = alerts.join('');

  const histWrap = document.getElementById('hist-wrap');
  if (summary.histogram && meta.num_instances > 1) {
    histWrap.classList.remove('hidden');
    drawHistogram(summary.histogram);
  } else {
    histWrap.classList.add('hidden');
  }

  await loadInstance(0);
}

async function loadInstance(idx) {
  if (!state.selectedFilename) return;
  const applyZoom = document.getElementById('chk-zoom').checked;
  const data = await fetchJSON(
    `/api/datasets/${encodeURIComponent(state.selectedFilename)}/instance/${idx}?zoom=${applyZoom}`
  );
  state.instance = data;

  document.getElementById('map-title').textContent = `${data.filename} — Bài toán #${data.problem_index}`;
  const zoomBadge = document.getElementById('zoom-badge');
  if (data.zoom_applied && data.zoom_ratio !== null) {
    zoomBadge.textContent = `Zoom ratio: ${data.zoom_ratio.toFixed(2)}`;
    zoomBadge.classList.remove('hidden');
  } else {
    zoomBadge.classList.add('hidden');
  }

  const statsEl = document.getElementById('instance-stats');
  if (data.has_demand) {
    statsEl.innerHTML = `Số node: <strong>${data.nodes.length}</strong> | Tổng demand: <strong>${data.total_demand.toFixed(3)}</strong> | Số xe tối thiểu ước tính: <strong>${data.min_vehicles_needed}</strong>`;
  } else {
    statsEl.innerHTML = `Số node: <strong>${data.nodes.length}</strong> (không có cột demand)`;
  }

  const tbody = document.querySelector('#nodes-table tbody');
  tbody.innerHTML = data.nodes
    .map(
      (n) => `
      <tr>
        <td>${n.id}</td>
        <td>${n.role}</td>
        <td>${n.x.toFixed(4)}</td>
        <td>${n.y.toFixed(4)}</td>
        <td>${n.demand !== null ? n.demand.toFixed(4) : '-'}</td>
      </tr>`
    )
    .join('');

  document.getElementById('cli-eam').textContent = data.cli_commands.eam_eval;
  document.getElementById('cli-cmma').textContent = data.cli_commands.cmma_run;

  drawMap();
}

function demandColor(d, dMin, dMax) {
  const t = dMax > dMin ? (d - dMin) / (dMax - dMin) : 0.5;
  const hue = Math.round(220 - t * 185);
  return `hsl(${hue}, 82%, 48%)`;
}

function drawMap(hoverId = null) {
  const canvas = document.getElementById('map-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.instance || state.instance.nodes.length === 0) return;

  const nodes = state.instance.nodes;
  const showIds = document.getElementById('chk-ids').checked;
  const showDemands = document.getElementById('chk-demands').checked;

  let xMin = Math.min(...nodes.map((n) => n.x));
  let xMax = Math.max(...nodes.map((n) => n.x));
  let yMin = Math.min(...nodes.map((n) => n.y));
  let yMax = Math.max(...nodes.map((n) => n.y));
  if (xMin >= 0 && xMax <= 1 && yMin >= 0 && yMax <= 1) {
    xMin = 0; xMax = 1; yMin = 0; yMax = 1;
  } else {
    const padX = (xMax - xMin || 1) * 0.06;
    const padY = (yMax - yMin || 1) * 0.06;
    xMin -= padX; xMax += padX; yMin -= padY; yMax += padY;
  }

  const margin = { left: 48, right: 24, top: 24, bottom: 40 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  const toPx = (x, y) => ({
    px: margin.left + ((x - xMin) / (xMax - xMin || 1)) * plotW,
    py: margin.top + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH,
  });

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '11px sans-serif';
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    const gx = margin.left + t * plotW;
    const gy = margin.top + (1 - t) * plotH;
    ctx.beginPath();
    ctx.moveTo(gx, margin.top);
    ctx.lineTo(gx, margin.top + plotH);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(margin.left, gy);
    ctx.lineTo(margin.left + plotW, gy);
    ctx.stroke();

    const xv = xMin + t * (xMax - xMin);
    const yv = yMin + t * (yMax - yMin);
    ctx.textAlign = 'center';
    ctx.fillText(xv.toFixed(2), gx, h - 16);
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(2), margin.left - 6, gy + 4);
  }

  ctx.strokeStyle = '#94a3b8';
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  const custDemands = nodes.slice(1).map((n) => n.demand ?? 0);
  const dMin = custDemands.length ? Math.min(...custDemands) : 0;
  const dMax = custDemands.length ? Math.max(...custDemands) : 1;

  state.plottedPoints = [];

  for (let i = 1; i < nodes.length; i++) {
    const n = nodes[i];
    const { px, py } = toPx(n.x, n.y);
    const d = n.demand ?? 0;
    const r = nodes.length > 150 ? 3.5 : 4.5 + (dMax > 0 ? (d / dMax) * 4.5 : 0);
    state.plottedPoints.push({ node: n, px, py, r: Math.max(6, r + 2) });

    ctx.beginPath();
    ctx.arc(px, py, n.id === hoverId ? r + 3 : r, 0, Math.PI * 2);
    ctx.fillStyle = n.demand !== null ? demandColor(d, dMin, dMax) : '#2563eb';
    ctx.fill();
    ctx.lineWidth = n.id === hoverId ? 2.2 : 1;
    ctx.strokeStyle = n.id === hoverId ? '#0f172a' : '#1e293b';
    ctx.stroke();

    if (showIds || showDemands) {
      const parts = [];
      if (showIds) parts.push(String(n.id));
      if (showDemands && n.demand !== null) parts.push(`(${n.demand.toFixed(2)})`);
      ctx.fillStyle = '#0f172a';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(parts.join(' '), px + r + 3, py - 3);
    }
  }

  const depot = nodes[0];
  const { px: dx, py: dy } = toPx(depot.x, depot.y);
  state.plottedPoints.push({ node: depot, px: dx, py: dy, r: 11 });
  drawStar(ctx, dx, dy, 5, 11, 5, '#dc2626', '#0f172a');
  if (showIds) {
    ctx.fillStyle = '#991b1b';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('0 (Depot)', dx + 12, dy - 4);
  }
}

function drawStar(ctx, cx, cy, spikes, outerR, innerR, fill, stroke) {
  let rot = (Math.PI / 2) * 3;
  const step = Math.PI / spikes;
  ctx.beginPath();
  ctx.moveTo(cx, cy - outerR);
  for (let i = 0; i < spikes; i++) {
    ctx.lineTo(cx + Math.cos(rot) * outerR, cy + Math.sin(rot) * outerR);
    rot += step;
    ctx.lineTo(cx + Math.cos(rot) * innerR, cy + Math.sin(rot) * innerR);
    rot += step;
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function drawHistogram(hist) {
  const canvas = document.getElementById('hist-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const counts = hist.counts;
  const edges = hist.edges;
  const maxC = Math.max(1, ...counts);
  const pad = { left: 32, right: 12, top: 10, bottom: 24 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const barW = plotW / counts.length;

  counts.forEach((c, i) => {
    const bh = (c / maxC) * plotH;
    const x = pad.left + i * barW + 1;
    const y = pad.top + plotH - bh;
    ctx.fillStyle = '#0ea5e9';
    ctx.fillRect(x, y, Math.max(2, barW - 2), bh);
  });

  ctx.fillStyle = '#64748b';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(edges[0].toFixed(2), pad.left, h - 6);
  ctx.textAlign = 'right';
  ctx.fillText(edges[edges.length - 1].toFixed(2), w - pad.right, h - 6);
}

// ==================== STEP 3: E-AM & CMMA RUNNERS ====================

function populateRunnerDatasetSelects() {
  const ds = state.datasets;
  const trSel = document.getElementById('tr-val-dataset');
  const evSel = document.getElementById('ev-val-dataset');
  const cmSel = document.getElementById('cm-load-file');

  const dsOpts = ds.map((d) => `<option value="${d.filename}">${d.filename} (${d.num_instances} bài, ${d.customer_size} khách)</option>`).join('');
  trSel.innerHTML = `<option value="auto" selected>Tự động theo số khách &amp; xe</option>${dsOpts}<option value="random">random (sinh ngẫu nhiên)</option>`;
  evSel.innerHTML = `${dsOpts}<option value="random">random</option>`;
  cmSel.innerHTML = dsOpts;

  if (ds.some((d) => d.filename === 'cvrp20-v4.npy')) {
    evSel.value = 'cvrp20-v4.npy';
    cmSel.value = 'cvrp20-v4.npy';
  }
  syncEvalDatasetDefaults();
  syncCmmaDatasetDefaults();
}

async function loadCheckpoints() {
  try {
    const data = await fetchJSON('/api/checkpoints');
    state.checkpoints = data.checkpoints;

    const resumeSel = document.getElementById('tr-resume');
    const evalSel = document.getElementById('ev-load-path');

    const ckptOpts = state.checkpoints
      .map((c) => `<option value="${c.path}">${c.path} (${formatBytes(c.size_bytes)})</option>`)
      .join('');
    resumeSel.innerHTML = `<option value="">Không resume (train từ đầu)</option>${ckptOpts}`;
    evalSel.innerHTML = state.checkpoints.length
      ? ckptOpts
      : '<option value="">Chưa có checkpoint nào trong saved_model/</option>';

    const pomoText = document.getElementById('pomo-status-text');
    pomoText.textContent = data.pomo_tsp20_ready
      ? 'POMO TSP20: Sẵn sàng cho mo_cvrp_re_tsp'
      : 'Lưu ý: chưa có checkpoint POMO TSP20 (chỉ cần nếu chạy *_re_tsp)';

    updateEamPreview();
  } catch (err) {
    console.error(err);
  }
}

function getTrainPayload() {
  return {
    problem: document.getElementById('tr-problem').value,
    customer_size: Number(document.getElementById('tr-customer-size').value),
    vehicle_num: Number(document.getElementById('tr-vehicle-num').value),
    batch_size: Number(document.getElementById('tr-batch-size').value),
    epoch_size: Number(document.getElementById('tr-epoch-size').value),
    epoch_end: Number(document.getElementById('tr-epoch-end').value),
    mo_reward_type: document.getElementById('tr-mo-reward').value,
    lr_model: Number(document.getElementById('tr-lr').value),
    val_dataset: document.getElementById('tr-val-dataset').value,
    val_range_start: Number(document.getElementById('tr-val-start').value),
    val_range_end: Number(document.getElementById('tr-val-end').value),
    val_batch_size: Number(document.getElementById('tr-val-batch').value),
    eval_type: document.getElementById('tr-eval-type').value,
    resume: document.getElementById('tr-resume').value,
    run_name: document.getElementById('tr-run-name').value,
    quick_eval: document.getElementById('tr-quick-eval').checked,
    no_eval_in_train: document.getElementById('tr-no-eval').checked,
    no_cuda: document.getElementById('tr-no-cuda').checked,
    no_log: document.getElementById('tr-no-log').checked,
    no_save: document.getElementById('tr-no-save').checked,
  };
}

function getEvalPayload() {
  return {
    load_path: document.getElementById('ev-load-path').value,
    problem: document.getElementById('ev-problem').value,
    customer_size: Number(document.getElementById('ev-customer-size').value),
    vehicle_num: Number(document.getElementById('ev-vehicle-num').value),
    val_dataset: document.getElementById('ev-val-dataset').value,
    val_customer_size: Number(document.getElementById('ev-val-customer-size').value),
    val_vehicle_num: Number(document.getElementById('ev-val-vehicle-num').value),
    val_range_start: Number(document.getElementById('ev-val-start').value),
    val_range_end: Number(document.getElementById('ev-val-end').value),
    val_batch_size: Number(document.getElementById('ev-val-batch').value),
    eval_type: document.getElementById('ev-eval-type').value,
    sample_times: Number(document.getElementById('ev-sample-times').value),
    max_parallel: Number(document.getElementById('ev-max-parallel').value),
    zoom: document.getElementById('ev-zoom').value,
    run_name: document.getElementById('ev-run-name').value,
    quick_eval: document.getElementById('ev-quick-eval').checked,
    no_cuda: document.getElementById('ev-no-cuda').checked,
    no_log: document.getElementById('ev-no-log').checked,
  };
}

function getCmmaPayload() {
  return {
    load_file: document.getElementById('cm-load-file').value,
    problem_index: Number(document.getElementById('cm-problem-index').value),
    vehicle_num: Number(document.getElementById('cm-vehicle-num').value),
    population_size: Number(document.getElementById('cm-pop-size').value),
    N_gen: Number(document.getElementById('cm-n-gen').value),
    save_per_gen: Number(document.getElementById('cm-save-per-gen').value),
    Iter: Number(document.getElementById('cm-iter').value),
    LS_iter: Number(document.getElementById('cm-ls-iter').value),
    ref_point_0: Number(document.getElementById('cm-ref-0').value),
    ref_point_1: Number(document.getElementById('cm-ref-1').value),
    use_ref_point: document.getElementById('cm-use-ref').checked,
    no_log: document.getElementById('cm-no-log').checked,
    seed: Number(document.getElementById('cm-seed').value),
    run_name: document.getElementById('cm-run-name').value,
  };
}

function suggestedMaxParallel(customerSize) {
  if (customerSize <= 20) return 16384;
  if (customerSize <= 50) return 4096;
  if (customerSize <= 100) return 2048;
  if (customerSize <= 200) return 512;
  return 128;
}

function syncEvalDatasetDefaults() {
  const filename = document.getElementById('ev-val-dataset').value;
  const meta = state.datasets.find((d) => d.filename === filename);
  if (!meta) return;
  document.getElementById('ev-val-customer-size').value = meta.customer_size;
  if (meta.vehicle_num) {
    document.getElementById('ev-val-vehicle-num').value = meta.vehicle_num;
  }
  document.getElementById('ev-val-start').value = 0;
  document.getElementById('ev-val-end').value = meta.num_instances;
  document.getElementById('ev-val-batch').value =
    meta.num_instances === 1 ? 1 : Math.min(100, meta.num_instances);
  document.getElementById('ev-max-parallel').value = suggestedMaxParallel(meta.customer_size);
}

function syncCmmaDatasetDefaults() {
  const filename = document.getElementById('cm-load-file').value;
  const meta = state.datasets.find((d) => d.filename === filename);
  if (!meta) return;
  const idxInput = document.getElementById('cm-problem-index');
  idxInput.max = Math.max(0, meta.num_instances - 1);
  if (Number(idxInput.value) >= meta.num_instances) idxInput.value = 0;
  if (meta.vehicle_num) {
    document.getElementById('cm-vehicle-num').value = meta.vehicle_num;
  }
  if (meta.ref_point && meta.ref_point.length === 2) {
    document.getElementById('cm-ref-0').value = meta.ref_point[0];
    document.getElementById('cm-ref-1').value = meta.ref_point[1];
  }
}

async function updateEamPreview() {
  const msgEl = document.getElementById('eam-validation-msg');
  const preEl = document.getElementById('eam-cmd-preview');
  const runBtn = document.getElementById('btn-run-eam');
  const url =
    state.eamMode === 'train'
      ? '/api/jobs/eam/train/preview'
      : '/api/jobs/eam/eval/preview';
  const payload = state.eamMode === 'train' ? getTrainPayload() : getEvalPayload();

  try {
    const res = await fetchJSON(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    preEl.textContent = res.command_preview;
    msgEl.innerHTML = '';
    runBtn.disabled = false;
  } catch (err) {
    preEl.textContent = '(Tham số chưa hợp lệ)';
    msgEl.innerHTML = `<div class="alert alert-warn">${err.message}</div>`;
    runBtn.disabled = true;
  }
}

async function updateCmmaPreview() {
  const msgEl = document.getElementById('cmma-validation-msg');
  const preEl = document.getElementById('cmma-cmd-preview');
  const runBtn = document.getElementById('btn-run-cmma');
  if (!document.getElementById('cm-load-file').value) return;
  try {
    const res = await fetchJSON('/api/jobs/cmma/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(getCmmaPayload()),
    });
    preEl.textContent = res.command_preview;
    msgEl.innerHTML = '';
    runBtn.disabled = false;
  } catch (err) {
    preEl.textContent = '(Tham số chưa hợp lệ)';
    msgEl.innerHTML = `<div class="alert alert-warn">${err.message}</div>`;
    runBtn.disabled = true;
  }
}

function statusPillClass(status) {
  return `pill pill-${status || 'idle'}`;
}

function statusLabel(status) {
  const map = {
    running: 'Đang chạy',
    completed: 'Hoàn thành',
    stopped: 'Đã dừng',
    failed: 'Lỗi',
  };
  return map[status] || status;
}

function renderJobLists() {
  const eamJobs = state.jobs.filter((j) => j.kind.startsWith('eam_'));
  const cmmaJobs = state.jobs.filter((j) => j.kind === 'cmma');

  const runningEam = eamJobs.filter((j) => j.status === 'running').length;
  const runningCmma = cmmaJobs.filter((j) => j.status === 'running').length;
  const badgeEam = document.getElementById('badge-running-eam');
  const badgeCmma = document.getElementById('badge-running-cmma');
  badgeEam.textContent = `${runningEam} chạy`;
  badgeEam.classList.toggle('hidden', runningEam === 0);
  badgeCmma.textContent = `${runningCmma} chạy`;
  badgeCmma.classList.toggle('hidden', runningCmma === 0);

  renderJobListInto('eam-jobs-list', eamJobs, state.activeEamJobId, 'eam');
  renderJobListInto('cmma-jobs-list', cmmaJobs, state.activeCmmaJobId, 'cmma');
  updateActiveJobControls('eam');
  updateActiveJobControls('cmma');
}

function renderJobListInto(containerId, jobs, activeId, group) {
  const el = document.getElementById(containerId);
  if (jobs.length === 0) {
    el.innerHTML = '<div class="muted">Chưa có job nào được chạy từ Web UI.</div>';
    return;
  }
  el.innerHTML = jobs
    .map(
      (j) => `
      <div class="job-item ${j.job_id === activeId ? 'selected' : ''}" data-job="${j.job_id}">
        <div class="job-top">
          <span>${j.title}</span>
          <span class="${statusPillClass(j.status)}">${statusLabel(j.status)}</span>
        </div>
        <div class="job-meta">ID: ${j.job_id} | PID: ${j.pid ?? '-'} | Bắt đầu: ${formatTime(j.started_at)}</div>
      </div>`
    )
    .join('');

  el.querySelectorAll('.job-item').forEach((item) => {
    item.addEventListener('click', () => selectJob(group, item.dataset.job));
  });
}

function updateActiveJobControls(group) {
  const jobId = group === 'eam' ? state.activeEamJobId : state.activeCmmaJobId;
  const job = state.jobs.find((j) => j.job_id === jobId);
  const stopBtn = document.getElementById(group === 'eam' ? 'btn-stop-eam' : 'btn-stop-cmma');
  const pill = document.getElementById(group === 'eam' ? 'eam-active-status' : 'cmma-active-status');
  if (!job) {
    stopBtn.disabled = true;
    pill.className = 'pill pill-idle';
    pill.textContent = 'Chưa chạy';
    return;
  }
  stopBtn.disabled = job.status !== 'running';
  pill.className = statusPillClass(job.status);
  pill.textContent = `${statusLabel(job.status)} (PID ${job.pid ?? '-'})`;
}

async function refreshJobs() {
  const res = await fetchJSON('/api/jobs');
  state.jobs = res.jobs;
  renderJobLists();
}

function updateProgressBanner(group, progressText, status) {
  const wrap = document.getElementById(group === 'eam' ? 'eam-progress-wrap' : 'cmma-progress-wrap');
  const fill = document.getElementById(group === 'eam' ? 'eam-progress-fill' : 'cmma-progress-fill');
  const textEl = document.getElementById(group === 'eam' ? 'eam-progress-text' : 'cmma-progress-text');
  if (!progressText) {
    wrap.classList.add('hidden');
    return;
  }
  wrap.classList.remove('hidden');
  textEl.textContent = progressText;
  const match = progressText.match(/(\d+)%\|/);
  if (match) {
    fill.style.width = `${Math.min(100, Number(match[1]))}%`;
  } else if (status === 'completed') {
    fill.style.width = '100%';
  }
}

async function selectJob(group, jobId) {
  if (group === 'eam') state.activeEamJobId = jobId;
  else state.activeCmmaJobId = jobId;
  renderJobLists();

  const consoleEl = document.getElementById(group === 'eam' ? 'eam-log-console' : 'cmma-log-console');
  const autoScrollEl = document.getElementById(group === 'eam' ? 'eam-autoscroll' : 'cmma-autoscroll');

  if (state.streams[group]) {
    state.streams[group].close();
    delete state.streams[group];
  }

  const detail = await fetchJSON(`/api/jobs/${jobId}`);
  const job = detail.job;
  consoleEl.textContent = (job.logs || []).join('\n');
  updateProgressBanner(group, job.progress, job.status);
  if (autoScrollEl.checked) {
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  if (job.status === 'running') {
    const offset = (job.logs || []).length;
    const es = new EventSource(`/api/jobs/${jobId}/stream?offset=${offset}`);
    state.streams[group] = es;

    es.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.progress !== undefined) {
        updateProgressBanner(group, msg.progress, msg.status);
      }
      if (msg.lines && msg.lines.length) {
        consoleEl.textContent += (consoleEl.textContent ? '\n' : '') + msg.lines.join('\n');
        if (autoScrollEl.checked) {
          consoleEl.scrollTop = consoleEl.scrollHeight;
        }
      }
    };
    es.addEventListener('done', async (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.progress !== undefined) {
          updateProgressBanner(group, msg.progress, msg.status);
        }
      } catch (_) {}
      es.close();
      delete state.streams[group];
      await refreshJobs();
      await Promise.all([loadCheckpoints(), loadSystemStatus(), loadResultsList()]);
    });
    es.onerror = () => {
      es.close();
      delete state.streams[group];
    };
  }
}

async function startEamJob() {
  const url = state.eamMode === 'train' ? '/api/jobs/eam/train' : '/api/jobs/eam/eval';
  const payload = state.eamMode === 'train' ? getTrainPayload() : getEvalPayload();
  const res = await fetchJSON(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  await refreshJobs();
  await selectJob('eam', res.job.job_id);
}

async function startCmmaJob() {
  const res = await fetchJSON('/api/jobs/cmma/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(getCmmaPayload()),
  });
  await refreshJobs();
  await selectJob('cmma', res.job.job_id);
}

async function stopActiveJob(group) {
  const jobId = group === 'eam' ? state.activeEamJobId : state.activeCmmaJobId;
  if (!jobId) return;
  await fetchJSON(`/api/jobs/${jobId}/stop`, { method: 'POST' });
  await refreshJobs();
}

async function refreshTensorBoardStatus() {
  const tb = await fetchJSON('/api/tensorboard/status');
  const pill = document.getElementById('tb-status-pill');
  const btn = document.getElementById('btn-tb-toggle');
  const link = document.getElementById('tb-link');

  if (tb.running) {
    pill.className = 'pill pill-completed';
    pill.textContent = `TensorBoard: Đang chạy (:${tb.port})`;
    btn.textContent = 'Tắt TensorBoard';
    link.href = tb.url;
    link.classList.remove('hidden');
  } else {
    pill.className = 'pill pill-idle';
    pill.textContent = 'TensorBoard: Đang tắt';
    btn.textContent = 'Bật TensorBoard';
    link.classList.add('hidden');
  }
  return tb.running;
}

function applyTrainPreset(name) {
  const presets = {
    train_quick: { cus: 20, veh: 4, bs: 256, es: 2560, ep: 3, val_end: 200, vbs: 100, quick: true, run: 'web_test_20_v4' },
    train_20: { cus: 20, veh: 4, bs: 512, es: 128000, ep: 200, val_end: 200, vbs: 100, quick: true, run: 'mo_cvrp_20_v4' },
    train_50: { cus: 50, veh: 7, bs: 256, es: 128000, ep: 200, val_end: 200, vbs: 100, quick: true, run: 'mo_cvrp_50_v7' },
    train_100: { cus: 100, veh: 11, bs: 64, es: 128000, ep: 200, val_end: 200, vbs: 50, quick: true, run: 'mo_cvrp_100_v11' },
  };
  const p = presets[name];
  if (!p) return;
  document.getElementById('tr-problem').value = 'mo_cvrp';
  document.getElementById('tr-customer-size').value = p.cus;
  document.getElementById('tr-vehicle-num').value = p.veh;
  document.getElementById('tr-batch-size').value = p.bs;
  document.getElementById('tr-epoch-size').value = p.es;
  document.getElementById('tr-epoch-end').value = p.ep;
  document.getElementById('tr-val-dataset').value = 'auto';
  document.getElementById('tr-val-start').value = 0;
  document.getElementById('tr-val-end').value = p.val_end;
  document.getElementById('tr-val-batch').value = p.vbs;
  document.getElementById('tr-quick-eval').checked = p.quick;
  document.getElementById('tr-run-name').value = p.run;
  updateEamPreview();
}

function applyCmmaPreset(name) {
  const presets = {
    quick: { pop: 50, gen: 200, save: 50, iter: 40, ls: 300 },
    medium: { pop: 100, gen: 1000, save: 100, iter: 40, ls: 300 },
    full: { pop: 100, gen: 10000, save: 100, iter: 40, ls: 300 },
  };
  const p = presets[name];
  if (!p) return;
  document.getElementById('cm-pop-size').value = p.pop;
  document.getElementById('cm-n-gen').value = p.gen;
  document.getElementById('cm-save-per-gen').value = p.save;
  document.getElementById('cm-iter').value = p.iter;
  document.getElementById('cm-ls-iter').value = p.ls;
  updateCmmaPreview();
}

// ==================== STEP 4: RESULTS, PARETO & ROUTE VISUALIZER ====================

const VEHICLE_COLORS = [
  '#2563eb', '#16a34a', '#f59e0b', '#9333ea',
  '#06b6d4', '#e11d48', '#84cc16', '#f97316',
  '#4f46e5', '#14b8a6', '#db2777', '#65a30d',
];

function vehicleColor(idx) {
  return VEHICLE_COLORS[idx % VEHICLE_COLORS.length];
}

async function loadResultsList(preferredId = null) {
  const data = await fetchJSON('/api/results');
  const all = [...(data.eam || []), ...(data.cmma || [])];
  state.resultsList = all;

  const emptyEl = document.getElementById('results-empty-notice');
  const ctrlWrap = document.getElementById('results-controls-wrap');
  const vizWrap = document.getElementById('res-viz-wrap');
  const rawCard = document.getElementById('res-raw-card');

  if (all.length === 0) {
    emptyEl.innerHTML = '<div class="alert alert-info">Chưa có kết quả đánh giá nào trong <code>saved_model/</code> (cần chạy <code>--eval_only</code>) hoặc <code>result/</code> (CMMA). Hãy sang Tab 2 hoặc Tab 3 để chạy thử trước!</div>';
    emptyEl.classList.remove('hidden');
    ctrlWrap.classList.add('hidden');
    vizWrap.classList.add('hidden');
    rawCard.classList.add('hidden');
    return;
  }

  emptyEl.classList.add('hidden');
  ctrlWrap.classList.remove('hidden');
  vizWrap.classList.remove('hidden');

  if ((data.eam || []).length === 0) {
    emptyEl.innerHTML = '<div class="alert alert-info">Hiện mới có kết quả <strong>CMMA</strong> trong <code>result/</code>. Để xem và so sánh thêm <strong>E-AM</strong>, hãy vào <strong>Tab 2 → Đánh giá (--eval_only)</strong> và chạy với một checkpoint <code>epoch-*.pt</code> có sẵn.</div>';
    emptyEl.classList.remove('hidden');
  }

  const selA = document.getElementById('res-select-a');
  const selB = document.getElementById('res-select-b');
  const curA = preferredId || selA.value || all[0].result_id;
  const curB = selB.value || '';

  const optsHtml = all
    .map((r) => `<option value="${r.result_id}">${r.title}${r.hv != null ? ` (HV=${Number(r.hv).toFixed(4)})` : ''}</option>`)
    .join('');

  selA.innerHTML = optsHtml;
  selA.value = all.some((r) => r.result_id === curA) ? curA : all[0].result_id;

  selB.innerHTML = `<option value="">-- Không so sánh --</option>${optsHtml}`;
  selB.value = all.some((r) => r.result_id === curB) ? curB : '';

  await onResultAChanged();
}

async function onResultAChanged() {
  const idA = document.getElementById('res-select-a').value;
  if (!idA) return;
  const summaryA = await fetchJSON(`/api/results/summary?result_id=${encodeURIComponent(idA)}`);
  state.resultSummaryA = summaryA;

  const genSelA = document.getElementById('res-gen-a');
  const instInputA = document.getElementById('res-inst-a');

  if (summaryA.kind === 'cmma' && summaryA.generations.length) {
    genSelA.innerHTML = summaryA.generations
      .map((g) => `<option value="${g}" ${g === summaryA.chosen_gen ? 'selected' : ''}>Gen ${g}</option>`)
      .join('');
    genSelA.disabled = false;
    instInputA.value = 0;
    instInputA.max = 0;
    instInputA.disabled = true;
  } else {
    genSelA.innerHTML = '<option value="">N/A</option>';
    genSelA.disabled = true;
    const maxInst = Math.max(0, (summaryA.num_instances || 1) - 1);
    instInputA.max = maxInst;
    if (Number(instInputA.value) > maxInst) instInputA.value = 0;
    instInputA.disabled = maxInst === 0;
  }

  await refreshResultBControls();
  await loadCompareAndVisualize();
}

async function refreshResultBControls() {
  const idB = document.getElementById('res-select-b').value;
  const sub = document.getElementById('res-b-subcontrols');
  if (!idB) {
    sub.classList.add('hidden');
    return;
  }
  sub.classList.remove('hidden');
  const summaryB = await fetchJSON(`/api/results/summary?result_id=${encodeURIComponent(idB)}`);
  const genSelB = document.getElementById('res-gen-b');
  const instInputB = document.getElementById('res-inst-b');

  if (summaryB.kind === 'cmma' && summaryB.generations.length) {
    genSelB.innerHTML = summaryB.generations
      .map((g) => `<option value="${g}" ${g === summaryB.chosen_gen ? 'selected' : ''}>Gen ${g}</option>`)
      .join('');
    genSelB.disabled = false;
    instInputB.value = 0;
    instInputB.max = 0;
    instInputB.disabled = true;
  } else {
    genSelB.innerHTML = '<option value="">N/A</option>';
    genSelB.disabled = true;
    const maxInst = Math.max(0, (summaryB.num_instances || 1) - 1);
    instInputB.max = maxInst;
    if (Number(instInputB.value) > maxInst) instInputB.value = 0;
    instInputB.disabled = maxInst === 0;
  }
}

async function loadCompareAndVisualize() {
  const idA = document.getElementById('res-select-a').value;
  if (!idA) return;
  const genA = document.getElementById('res-gen-a').value;
  const instA = Number(document.getElementById('res-inst-a').value) || 0;
  const idB = document.getElementById('res-select-b').value;
  const genB = document.getElementById('res-gen-b').value;
  const instB = Number(document.getElementById('res-inst-b').value) || 0;

  const summaryA = await fetchJSON(
    `/api/results/summary?result_id=${encodeURIComponent(idA)}${genA ? `&gen=${genA}` : ''}`
  );
  state.resultSummaryA = summaryA;

  const qs = new URLSearchParams({ result_a: idA, inst_a: String(instA) });
  if (genA) qs.set('gen_a', genA);
  if (idB) {
    qs.set('result_b', idB);
    qs.set('inst_b', String(instB));
    if (genB) qs.set('gen_b', genB);
  }

  const cmp = await fetchJSON(`/api/results/compare?${qs.toString()}`);
  state.compareData = cmp;

  const pA = cmp.primary;
  const pB = cmp.secondary;

  // Render metrics cards
  const metricsEl = document.getElementById('res-metrics-grid');
  const hvShow = pA.hv_ratio != null ? pA.hv_ratio.toFixed(4) : (summaryA.hv != null ? Number(summaryA.hv).toFixed(4) : 'N/A');
  metricsEl.innerHTML = `
    <div class="metric"><div class="metric-label">Dataset</div><div class="metric-value" style="font-size:14px">${pA.dataset || 'N/A'}</div><div class="metric-sub">Bài toán #${pA.dataset_problem_index}</div></div>
    <div class="metric"><div class="metric-label">Hypervolume (HV)</div><div class="metric-value">${hvShow}</div><div class="metric-sub">Ref: [${(pA.ref_point || []).join(', ')}]</div></div>
    <div class="metric"><div class="metric-label">Min-Sum tốt nhất (f₁)</div><div class="metric-value">${pA.best_sum != null ? pA.best_sum.toFixed(4) : 'N/A'}</div></div>
    <div class="metric"><div class="metric-label">Min-Max tốt nhất (f₂)</div><div class="metric-value">${pA.best_max != null ? pA.best_max.toFixed(4) : 'N/A'}</div></div>
    <div class="metric"><div class="metric-label">Số nghiệm không bị trội</div><div class="metric-value">${pA.nondominated_count} / ${pA.points.length}</div></div>
  `;

  // Compare table if B is selected
  const cmpWrap = document.getElementById('res-compare-table-wrap');
  const legendB = document.getElementById('legend-b');
  if (pB) {
    legendB.classList.remove('hidden');
    cmpWrap.classList.remove('hidden');
    const hvA = pA.shared_hv ?? pA.hv_ratio ?? 0;
    const hvB = pB.shared_hv ?? pB.hv_ratio ?? 0;
    cmpWrap.innerHTML = `
      <table>
        <thead>
          <tr><th>Kết quả</th><th>Dataset / Bài</th><th>HV (cùng Ref)</th><th>Best Min-Sum (f₁)</th><th>Best Min-Max (f₂)</th><th>Nghiệm không bị trội</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>A (${pA.kind.toUpperCase()})</strong></td><td>${pA.dataset || '-'} #${pA.dataset_problem_index}</td><td>${hvA.toFixed(4)}</td><td>${pA.best_sum?.toFixed(4) ?? '-'}</td><td>${pA.best_max?.toFixed(4) ?? '-'}</td><td>${pA.nondominated_count} / ${pA.points.length}</td></tr>
          <tr><td><strong>B (${pB.kind.toUpperCase()})</strong></td><td>${pB.dataset || '-'} #${pB.dataset_problem_index}</td><td>${hvB.toFixed(4)}</td><td>${pB.best_sum?.toFixed(4) ?? '-'}</td><td>${pB.best_max?.toFixed(4) ?? '-'}</td><td>${pB.nondominated_count} / ${pB.points.length}</td></tr>
        </tbody>
      </table>`;
  } else {
    legendB.classList.add('hidden');
    cmpWrap.classList.add('hidden');
  }

  // Populate solution selector (unique / non-dominated first)
  const solSel = document.getElementById('res-sol-select');
  const sortedPts = [...pA.points].sort((a, b) => a.f1 - b.f1);
  solSel.innerHTML = sortedPts
    .map(
      (pt) =>
        `<option value="${pt.sol_idx}">#${pt.sol_idx} | f₁=${pt.f1.toFixed(3)}, f₂=${pt.f2.toFixed(3)}${pt.is_nondominated ? ' ★' : ''}</option>`
    )
    .join('');

  state.selectedSolIdx = pA.knee_idx ?? pA.min_sum_idx ?? 0;
  solSel.value = String(state.selectedSolIdx);
  state.highlightedVehicleId = null;
  document.getElementById('btn-clear-veh-highlight').classList.add('hidden');

  // HV history chart (CMMA)
  const hvWrap = document.getElementById('hv-history-wrap');
  if (summaryA.hv_history && summaryA.hv_history.length > 1) {
    hvWrap.classList.remove('hidden');
    drawHvHistory(summaryA.hv_history);
  } else {
    hvWrap.classList.add('hidden');
  }

  // Raw log / eval.txt
  const rawCard = document.getElementById('res-raw-card');
  if (summaryA.raw_text) {
    rawCard.classList.remove('hidden');
    document.getElementById('res-raw-text').textContent = summaryA.raw_text;
  } else {
    rawCard.classList.add('hidden');
  }

  drawParetoChart();
  await loadSolutionRoutes(state.selectedSolIdx);
}

async function loadSolutionRoutes(solIdx) {
  const idA = document.getElementById('res-select-a').value;
  if (!idA) return;
  const genA = document.getElementById('res-gen-a').value;
  const instA = Number(document.getElementById('res-inst-a').value) || 0;

  state.selectedSolIdx = solIdx;
  document.getElementById('res-sol-select').value = String(solIdx);
  drawParetoChart();

  const qs = new URLSearchParams({
    result_id: idA,
    instance_offset: String(instA),
    sol_idx: String(solIdx),
  });
  if (genA) qs.set('gen', genA);

  const summaryBar = document.getElementById('route-summary-bar');
  try {
    const res = await fetchJSON(`/api/results/routes?${qs.toString()}`);
    state.routeData = res;
    document.getElementById('route-map-title').textContent = `Bản đồ Lộ trình — Nghiệm #${solIdx} (Bài #${res.problem_index})`;
    summaryBar.innerHTML =
      `Tổng quãng đường (f₁): <strong>${res.objective_sum.toFixed(4)}</strong> | ` +
      `Xe dài nhất (f₂): <strong>${res.objective_max.toFixed(4)}</strong> | ` +
      `Chênh lệch max-min: <strong>${res.balance_gap.toFixed(4)}</strong>`;

    renderVehiclesTable(res);
    drawRouteMap();
  } catch (err) {
    state.routeData = null;
    summaryBar.innerHTML = `<div class="alert alert-warn">${err.message}</div>`;
    document.querySelector('#vehicles-table tbody').innerHTML = '';
    const ctx = document.getElementById('route-canvas').getContext('2d');
    ctx.clearRect(0, 0, 600, 440);
  }
}

function renderVehiclesTable(routeData) {
  const tbody = document.querySelector('#vehicles-table tbody');
  const maxDist = Math.max(1e-6, routeData.objective_max);

  tbody.innerHTML = routeData.vehicles
    .map((v, idx) => {
      const color = vehicleColor(idx);
      const pct = Math.min(100, (v.distance / maxDist) * 100);
      const selected = state.highlightedVehicleId === v.vehicle_id;
      return `
        <tr data-veh="${v.vehicle_id}" class="${selected ? 'selected' : ''}">
          <td><span class="veh-badge"><i class="veh-swatch" style="background:${color}"></i>Xe ${v.vehicle_id}</span></td>
          <td>${v.customer_count}</td>
          <td>${v.distance.toFixed(4)}</td>
          <td>
            <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%; background:${color}"></div></div>
          </td>
          <td>${v.demand.toFixed(3)}</td>
          <td><code>${v.route.join(' → ')}</code></td>
        </tr>`;
    })
    .join('');

  tbody.querySelectorAll('tr').forEach((row) => {
    row.addEventListener('click', () => {
      const vid = Number(row.dataset.veh);
      state.highlightedVehicleId = state.highlightedVehicleId === vid ? null : vid;
      document
        .getElementById('btn-clear-veh-highlight')
        .classList.toggle('hidden', state.highlightedVehicleId === null);
      renderVehiclesTable(routeData);
      drawRouteMap();
    });
  });
}

function drawParetoChart(hoverPt = null) {
  const canvas = document.getElementById('pareto-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.compareData || !state.compareData.primary) return;
  const onlyNd = document.getElementById('chk-only-nd').checked;

  const filterPts = (pts) => (onlyNd ? pts.filter((p) => p.is_nondominated) : pts);
  const ptsA = filterPts(state.compareData.primary.points || []);
  const ptsB = state.compareData.secondary ? filterPts(state.compareData.secondary.points || []) : [];

  const allPts = [...ptsA, ...ptsB];
  if (allPts.length === 0) return;

  let xMin = Math.min(...allPts.map((p) => p.f1));
  let xMax = Math.max(...allPts.map((p) => p.f1));
  let yMin = Math.min(...allPts.map((p) => p.f2));
  let yMax = Math.max(...allPts.map((p) => p.f2));
  const padX = (xMax - xMin || 1) * 0.1;
  const padY = (yMax - yMin || 1) * 0.12;
  xMin -= padX; xMax += padX; yMin -= padY; yMax += padY;

  const margin = { left: 56, right: 20, top: 20, bottom: 42 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  const toPx = (f1, f2) => ({
    px: margin.left + ((f1 - xMin) / (xMax - xMin || 1)) * plotW,
    py: margin.top + plotH - ((f2 - yMin) / (yMax - yMin || 1)) * plotH,
  });

  // Grid & axes
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '11px sans-serif';
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    const gx = margin.left + t * plotW;
    const gy = margin.top + (1 - t) * plotH;
    ctx.beginPath();
    ctx.moveTo(gx, margin.top);
    ctx.lineTo(gx, margin.top + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(margin.left, gy);
    ctx.lineTo(margin.left + plotW, gy);
    ctx.stroke();

    ctx.textAlign = 'center';
    ctx.fillText((xMin + t * (xMax - xMin)).toFixed(2), gx, h - 22);
    ctx.textAlign = 'right';
    ctx.fillText((yMin + t * (yMax - yMin)).toFixed(2), margin.left - 6, gy + 4);
  }
  ctx.strokeStyle = '#94a3b8';
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);
  ctx.fillStyle = '#334155';
  ctx.font = 'bold 11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Tổng quãng đường (f₁ - Min-Sum)', margin.left + plotW / 2, h - 5);

  const drawFrontLine = (pts, color) => {
    const nd = pts.filter((p) => p.is_nondominated).sort((a, b) => a.f1 - b.f1);
    if (nd.length < 2) return;
    ctx.beginPath();
    nd.forEach((p, i) => {
      const { px, py } = toPx(p.f1, p.f2);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  };

  if (ptsB.length) drawFrontLine(ptsB, 'rgba(249, 115, 22, 0.55)');
  drawFrontLine(ptsA, 'rgba(37, 99, 235, 0.55)');

  // Draw B points
  ptsB.forEach((p) => {
    const { px, py } = toPx(p.f1, p.f2);
    ctx.beginPath();
    ctx.arc(px, py, p.is_nondominated ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = p.is_nondominated ? '#f97316' : 'rgba(249, 115, 22, 0.45)';
    ctx.fill();
  });

  // Draw A points & save hitboxes
  state.paretoPlottedPoints = [];
  ptsA.forEach((p) => {
    const { px, py } = toPx(p.f1, p.f2);
    const isSelected = p.sol_idx === state.selectedSolIdx;
    const isHover = hoverPt && hoverPt.sol_idx === p.sol_idx;
    state.paretoPlottedPoints.push({ pt: p, px, py });

    ctx.beginPath();
    ctx.arc(px, py, isSelected ? 7.5 : isHover ? 6.5 : p.is_nondominated ? 5 : 3.2, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? '#dc2626' : p.is_nondominated ? '#2563eb' : 'rgba(37, 99, 235, 0.35)';
    ctx.fill();
    if (isSelected || isHover) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#0f172a';
      ctx.stroke();
    }
  });
}

function drawArrowhead(ctx, x1, y1, x2, y2, color) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 16) return;
  const mx = x1 + dx * 0.62;
  const my = y1 + dy * 0.62;
  const angle = Math.atan2(dy, dx);
  const size = 7;
  ctx.beginPath();
  ctx.moveTo(mx, my);
  ctx.lineTo(mx - size * Math.cos(angle - Math.PI / 6), my - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(mx - size * Math.cos(angle + Math.PI / 6), my - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawRouteMap() {
  const canvas = document.getElementById('route-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.routeData || !state.routeData.nodes.length) return;

  const nodes = state.routeData.nodes;
  const showIds = document.getElementById('chk-route-ids').checked;
  const highlightId = state.highlightedVehicleId;

  let xMin = Math.min(...nodes.map((n) => n.x));
  let xMax = Math.max(...nodes.map((n) => n.x));
  let yMin = Math.min(...nodes.map((n) => n.y));
  let yMax = Math.max(...nodes.map((n) => n.y));
  if (xMin >= 0 && xMax <= 1 && yMin >= 0 && yMax <= 1) {
    xMin = 0; xMax = 1; yMin = 0; yMax = 1;
  } else {
    const padX = (xMax - xMin || 1) * 0.08;
    const padY = (yMax - yMin || 1) * 0.08;
    xMin -= padX; xMax += padX; yMin -= padY; yMax += padY;
  }

  const margin = { left: 46, right: 20, top: 20, bottom: 36 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  const toPx = (x, y) => ({
    px: margin.left + ((x - xMin) / (xMax - xMin || 1)) * plotW,
    py: margin.top + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH,
  });

  // Grid
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '10px sans-serif';
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    const gx = margin.left + t * plotW;
    const gy = margin.top + (1 - t) * plotH;
    ctx.beginPath(); ctx.moveTo(gx, margin.top); ctx.lineTo(gx, margin.top + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(margin.left, gy); ctx.lineTo(margin.left + plotW, gy); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText((xMin + t * (xMax - xMin)).toFixed(2), gx, h - 14);
    ctx.textAlign = 'right'; ctx.fillText((yMin + t * (yMax - yMin)).toFixed(2), margin.left - 6, gy + 3);
  }
  ctx.strokeStyle = '#94a3b8';
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Draw vehicle routes
  state.routeData.vehicles.forEach((v, idx) => {
    if (v.route.length <= 2) return;
    const isDimmed = highlightId !== null && highlightId !== v.vehicle_id;
    const color = vehicleColor(idx);
    ctx.save();
    ctx.globalAlpha = isDimmed ? 0.14 : 0.92;
    ctx.strokeStyle = color;
    ctx.lineWidth = highlightId === v.vehicle_id ? 3.2 : 2.1;

    ctx.beginPath();
    v.route.forEach((nid, stepIdx) => {
      const n = nodes[nid];
      const { px, py } = toPx(n.x, n.y);
      if (stepIdx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    if (!isDimmed) {
      for (let i = 0; i < v.route.length - 1; i++) {
        const a = toPx(nodes[v.route[i]].x, nodes[v.route[i]].y);
        const b = toPx(nodes[v.route[i + 1]].x, nodes[v.route[i + 1]].y);
        drawArrowhead(ctx, a.px, a.py, b.px, b.py, color);
      }
    }
    ctx.restore();
  });

  // Draw customer nodes colored by owning vehicle
  const nodeColorMap = {};
  state.routeData.vehicles.forEach((v, idx) => {
    v.route.forEach((nid) => {
      if (nid !== 0) nodeColorMap[nid] = vehicleColor(idx);
    });
  });

  for (let i = 1; i < nodes.length; i++) {
    const n = nodes[i];
    const { px, py } = toPx(n.x, n.y);
    const c = nodeColorMap[n.id] || '#64748b';
    const r = nodes.length > 150 ? 3.5 : 5;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = c;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#0f172a';
    ctx.stroke();

    if (showIds) {
      ctx.fillStyle = '#0f172a';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(String(n.id), px + r + 3, py - 2);
    }
  }

  // Depot
  const depot = toPx(nodes[0].x, nodes[0].y);
  drawStar(ctx, depot.px, depot.py, 5, 11, 5, '#dc2626', '#0f172a');
}

function drawHvHistory(history) {
  const canvas = document.getElementById('hv-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const valid = history.filter((pt) => pt.hv !== null && pt.hv !== undefined);
  if (valid.length < 2) return;

  const gMin = valid[0].gen;
  const gMax = valid[valid.length - 1].gen;
  let hMin = Math.min(...valid.map((p) => p.hv));
  let hMax = Math.max(...valid.map((p) => p.hv));
  const padY = (hMax - hMin || 0.01) * 0.15;
  hMin -= padY; hMax += padY;

  const pad = { left: 48, right: 16, top: 12, bottom: 24 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.strokeStyle = '#e2e8f0';
  ctx.strokeRect(pad.left, pad.top, plotW, plotH);

  ctx.beginPath();
  valid.forEach((pt, idx) => {
    const px = pad.left + ((pt.gen - gMin) / (gMax - gMin || 1)) * plotW;
    const py = pad.top + plotH - ((pt.hv - hMin) / (hMax - hMin || 1)) * plotH;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = '#16a34a';
  ctx.lineWidth = 2;
  ctx.stroke();

  valid.forEach((pt) => {
    const px = pad.left + ((pt.gen - gMin) / (gMax - gMin || 1)) * plotW;
    const py = pad.top + plotH - ((pt.hv - hMin) / (hMax - hMin || 1)) * plotH;
    ctx.beginPath();
    ctx.arc(px, py, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = '#16a34a';
    ctx.fill();
  });

  ctx.fillStyle = '#64748b';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`Gen ${gMin}`, pad.left, h - 6);
  ctx.textAlign = 'right';
  ctx.fillText(`Gen ${gMax} (HV=${valid[valid.length - 1].hv.toFixed(4)})`, w - pad.right, h - 6);
}

function setupParetoInteraction() {
  const canvas = document.getElementById('pareto-canvas');
  const hoverEl = document.getElementById('pareto-hover-info');

  const findNearest = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    let best = null;
    let bestDist = Infinity;
    for (const item of state.paretoPlottedPoints) {
      const dist = Math.hypot(mx - item.px, my - item.py);
      if (dist <= 11 && dist < bestDist) {
        best = item.pt;
        bestDist = dist;
      }
    }
    return best;
  };

  canvas.addEventListener('mousemove', (e) => {
    const hit = findNearest(e);
    if (hit) {
      hoverEl.textContent = `Nghiệm #${hit.sol_idx} — f₁: ${hit.f1.toFixed(4)}, f₂: ${hit.f2.toFixed(4)} (click để xem lộ trình)`;
      drawParetoChart(hit);
    } else {
      hoverEl.textContent = 'Click vào một điểm của A trên đồ thị để vẽ lộ trình xe';
      drawParetoChart(null);
    }
  });

  canvas.addEventListener('click', (e) => {
    const hit = findNearest(e);
    if (hit) loadSolutionRoutes(hit.sol_idx);
  });
}

function setupCanvasHover() {
  const canvas = document.getElementById('map-canvas');
  const hoverInfo = document.getElementById('hover-info');

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    let best = null;
    let bestDist = Infinity;
    for (const pt of state.plottedPoints) {
      const dist = Math.hypot(mx - pt.px, my - pt.py);
      if (dist <= pt.r && dist < bestDist) {
        best = pt.node;
        bestDist = dist;
      }
    }

    if (best) {
      const dem = best.demand !== null ? ` | Demand: ${best.demand.toFixed(4)}` : '';
      hoverInfo.textContent = `Node ${best.id} (${best.role}) — X: ${best.x.toFixed(4)}, Y: ${best.y.toFixed(4)}${dem}`;
      drawMap(best.id);
    } else {
      hoverInfo.textContent = 'Rê chuột lên điểm để xem chi tiết';
      drawMap(null);
    }
  });
}

function setupControls() {
  document.getElementById('btn-refresh-status').addEventListener('click', loadSystemStatus);
  document.getElementById('btn-refresh-datasets').addEventListener('click', () => loadDatasets());

  document.getElementById('btn-toggle-manage').addEventListener('click', () => {
    document.getElementById('manage-panel').classList.toggle('hidden');
  });

  document.getElementById('select-dataset').addEventListener('change', (e) => {
    selectDataset(e.target.value);
  });

  const slider = document.getElementById('instance-slider');
  const numInput = document.getElementById('instance-index');
  slider.addEventListener('input', () => {
    numInput.value = slider.value;
    loadInstance(Number(slider.value));
  });
  numInput.addEventListener('change', () => {
    const val = Math.max(0, Math.min(Number(numInput.max), Number(numInput.value) || 0));
    numInput.value = val;
    slider.value = val;
    loadInstance(val);
  });

  document.getElementById('chk-zoom').addEventListener('change', () => {
    loadInstance(Number(numInput.value));
  });
  document.getElementById('chk-ids').addEventListener('change', () => drawMap());
  document.getElementById('chk-demands').addEventListener('change', () => drawMap());

  document.getElementById('gen-customer-size').addEventListener('change', (e) => {
    const sz = e.target.value;
    const veh = document.getElementById('gen-vehicle-num').value;
    document.getElementById('gen-filename').value = `custom_cvrp${sz}-v${veh}.npy`;
  });
  document.getElementById('gen-vehicle-num').addEventListener('change', () => {
    const sz = document.getElementById('gen-customer-size').value;
    const veh = document.getElementById('gen-vehicle-num').value;
    document.getElementById('gen-filename').value = `custom_cvrp${sz}-v${veh}.npy`;
  });

  document.getElementById('btn-upload').addEventListener('click', async () => {
    const fileInput = document.getElementById('upload-file');
    const fb = document.getElementById('manage-feedback');
    if (!fileInput.files.length) {
      fb.innerHTML = '<div class="alert alert-warn">Vui lòng chọn một file .npy trước.</div>';
      return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
      const res = await fetchJSON('/api/datasets/upload', { method: 'POST', body: formData });
      fb.innerHTML = `<div class="alert alert-success">Đã tải lên <strong>${res.dataset.filename}</strong> thành công!</div>`;
      await loadDatasets(res.dataset.filename);
      await loadSystemStatus();
    } catch (err) {
      fb.innerHTML = `<div class="alert alert-error">Lỗi tải lên: ${err.message}</div>`;
    }
  });

  document.getElementById('btn-generate').addEventListener('click', async () => {
    const fb = document.getElementById('manage-feedback');
    const payload = {
      customer_size: Number(document.getElementById('gen-customer-size').value),
      num_instances: Number(document.getElementById('gen-num-instances').value),
      vehicle_num: Number(document.getElementById('gen-vehicle-num').value),
      vehicle_capacity: Number(document.getElementById('gen-capacity').value),
      seed: Number(document.getElementById('gen-seed').value),
      filename: document.getElementById('gen-filename').value,
    };
    try {
      const res = await fetchJSON('/api/datasets/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      fb.innerHTML = `<div class="alert alert-success">Đã sinh và lưu <strong>${res.dataset.filename}</strong>!</div>`;
      await loadDatasets(res.dataset.filename);
      await loadSystemStatus();
    } catch (err) {
      fb.innerHTML = `<div class="alert alert-error">Lỗi sinh dữ liệu: ${err.message}</div>`;
    }
  });

  // Step 3: E-AM & CMMA events
  document.getElementById('btn-mode-train').addEventListener('click', () => {
    state.eamMode = 'train';
    document.getElementById('btn-mode-train').classList.add('active');
    document.getElementById('btn-mode-eval').classList.remove('active');
    document.getElementById('eam-train-panel').classList.remove('hidden');
    document.getElementById('eam-eval-panel').classList.add('hidden');
    updateEamPreview();
  });

  document.getElementById('btn-mode-eval').addEventListener('click', () => {
    state.eamMode = 'eval';
    document.getElementById('btn-mode-eval').classList.add('active');
    document.getElementById('btn-mode-train').classList.remove('active');
    document.getElementById('eam-eval-panel').classList.remove('hidden');
    document.getElementById('eam-train-panel').classList.add('hidden');
    updateEamPreview();
  });

  document.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => applyTrainPreset(btn.dataset.preset));
  });
  document.querySelectorAll('[data-cmma-preset]').forEach((btn) => {
    btn.addEventListener('click', () => applyCmmaPreset(btn.dataset.cmmaPreset));
  });

  document.getElementById('tr-customer-size').addEventListener('change', (e) => {
    const defV = state.defaultVehicles[e.target.value];
    if (defV) document.getElementById('tr-vehicle-num').value = defV;
  });

  document.getElementById('ev-val-dataset').addEventListener('change', () => {
    syncEvalDatasetDefaults();
    updateEamPreview();
  });
  document.getElementById('cm-load-file').addEventListener('change', () => {
    syncCmmaDatasetDefaults();
    updateCmmaPreview();
  });

  document.querySelectorAll('#eam-train-panel input, #eam-train-panel select, #eam-eval-panel input, #eam-eval-panel select').forEach((el) => {
    el.addEventListener('change', updateEamPreview);
    el.addEventListener('input', updateEamPreview);
  });
  document.querySelectorAll('#tab-cmma input, #tab-cmma select').forEach((el) => {
    if (el.id === 'cmma-autoscroll') return;
    el.addEventListener('change', updateCmmaPreview);
    el.addEventListener('input', updateCmmaPreview);
  });

  document.getElementById('btn-refresh-ckpts').addEventListener('click', loadCheckpoints);
  document.getElementById('btn-run-eam').addEventListener('click', startEamJob);
  document.getElementById('btn-stop-eam').addEventListener('click', () => stopActiveJob('eam'));
  document.getElementById('btn-run-cmma').addEventListener('click', startCmmaJob);
  document.getElementById('btn-stop-cmma').addEventListener('click', () => stopActiveJob('cmma'));
  document.getElementById('btn-refresh-jobs-eam').addEventListener('click', refreshJobs);
  document.getElementById('btn-refresh-jobs-cmma').addEventListener('click', refreshJobs);
  document.getElementById('btn-clear-eam-log').addEventListener('click', () => {
    document.getElementById('eam-log-console').textContent = '';
  });
  document.getElementById('btn-clear-cmma-log').addEventListener('click', () => {
    document.getElementById('cmma-log-console').textContent = '';
  });

  document.getElementById('btn-tb-toggle').addEventListener('click', async () => {
    const running = await refreshTensorBoardStatus();
    if (running) {
      await fetchJSON('/api/tensorboard/stop', { method: 'POST' });
    } else {
      await fetchJSON('/api/tensorboard/start', { method: 'POST' });
    }
    await refreshTensorBoardStatus();
  });

  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const text = document.getElementById(btn.dataset.target).textContent;
      await navigator.clipboard.writeText(text);
      const old = btn.textContent;
      btn.textContent = 'Đã copy!';
      setTimeout(() => (btn.textContent = old), 1200);
    });
  });

  // Step 4: Results & Route visualizer events
  document.getElementById('btn-refresh-results').addEventListener('click', () => loadResultsList());
  document.getElementById('res-select-a').addEventListener('change', onResultAChanged);
  document.getElementById('res-gen-a').addEventListener('change', loadCompareAndVisualize);
  document.getElementById('res-inst-a').addEventListener('change', loadCompareAndVisualize);
  document.getElementById('res-select-b').addEventListener('change', async () => {
    await refreshResultBControls();
    await loadCompareAndVisualize();
  });
  document.getElementById('res-gen-b').addEventListener('change', loadCompareAndVisualize);
  document.getElementById('res-inst-b').addEventListener('change', loadCompareAndVisualize);

  document.getElementById('chk-only-nd').addEventListener('change', () => drawParetoChart());
  document.getElementById('chk-route-ids').addEventListener('change', drawRouteMap);
  document.getElementById('btn-clear-veh-highlight').addEventListener('click', () => {
    state.highlightedVehicleId = null;
    document.getElementById('btn-clear-veh-highlight').classList.add('hidden');
    if (state.routeData) renderVehiclesTable(state.routeData);
    drawRouteMap();
  });

  document.getElementById('res-sol-select').addEventListener('change', (e) => {
    loadSolutionRoutes(Number(e.target.value));
  });
  document.getElementById('btn-pick-min-sum').addEventListener('click', () => {
    const idx = state.compareData?.primary?.min_sum_idx;
    if (idx !== null && idx !== undefined) loadSolutionRoutes(idx);
  });
  document.getElementById('btn-pick-min-max').addEventListener('click', () => {
    const idx = state.compareData?.primary?.min_max_idx;
    if (idx !== null && idx !== undefined) loadSolutionRoutes(idx);
  });
  document.getElementById('btn-pick-knee').addEventListener('click', () => {
    const idx = state.compareData?.primary?.knee_idx;
    if (idx !== null && idx !== undefined) loadSolutionRoutes(idx);
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupControls();
  setupCanvasHover();
  setupParetoInteraction();
  await Promise.all([
    loadSystemStatus(),
    loadDatasets(),
    loadCheckpoints(),
    refreshJobs(),
    refreshTensorBoardStatus(),
    loadResultsList(),
  ]);
  updateEamPreview();
  updateCmmaPreview();
});
