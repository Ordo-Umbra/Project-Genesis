/* eslint-disable no-undef */
/**
 * Project Genesis – WebSocket-based Three.js viewer.
 *
 * Connects to the simulation server, requests world state & chunk data,
 * and renders voxels with semi-transparent materials.
 */

// ---------- Configuration ----------
const WS_URL = `ws://${location.hostname || "localhost"}:8765`;
const POLL_INTERVAL_MS = 1000; // how often to request state updates

// Band colours (void = invisible)
const BAND_COLORS = [
  null,                    // 0 – Void (not rendered)
  { color: 0x88ccff, opacity: 0.15 }, // 1 – Air
  { color: 0x8b6914, opacity: 0.6 },  // 2 – Soil
  { color: 0x888888, opacity: 0.85 }, // 3 – Stone
  { color: 0x333333, opacity: 1.0 },  // 4 – Bedrock
];

// ---------- Three.js Setup ----------
const canvas = document.getElementById("render-canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(canvas.clientWidth, canvas.clientHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
camera.position.set(50, 50, 50);

const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true;

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(40, 60, 30);
scene.add(dirLight);

// Voxel geometry (shared)
const boxGeo = new THREE.BoxGeometry(1, 1, 1);

// Materials (one per band)
const materials = BAND_COLORS.map((cfg) => {
  if (!cfg) return null;
  return new THREE.MeshLambertMaterial({
    color: cfg.color,
    transparent: cfg.opacity < 1,
    opacity: cfg.opacity,
  });
});

let voxelGroup = new THREE.Group();
scene.add(voxelGroup);

// ---------- Chart.js Setup ----------
const chartCtx = document.getElementById("s-chart").getContext("2d");
const sChart = new Chart(chartCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "S increment",
        data: [],
        borderColor: "#0f0",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { display: false },
      y: { ticks: { color: "#888", font: { size: 10 } }, grid: { color: "#333" } },
    },
    plugins: { legend: { display: false } },
  },
});

const MAX_CHART_POINTS = 200;

function pushChartPoint(step, sIncrement) {
  sChart.data.labels.push(step);
  sChart.data.datasets[0].data.push(sIncrement);
  if (sChart.data.labels.length > MAX_CHART_POINTS) {
    sChart.data.labels.shift();
    sChart.data.datasets[0].data.shift();
  }
  sChart.update();
}

// ---------- UI ----------
const infoEl = document.getElementById("info");
const speedSlider = document.getElementById("speed-slider");
let paused = false;

document.getElementById("btn-pause").addEventListener("click", () => { paused = true; });
document.getElementById("btn-play").addEventListener("click", () => { paused = false; });

// ---------- WebSocket ----------
let ws = null;
let worldDims = [32, 32, 32];

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    infoEl.textContent = "Connected";
    requestState();
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      handleMessage(msg);
    } catch { /* ignore non-JSON */ }
  };

  ws.onclose = () => {
    infoEl.textContent = "Disconnected – reconnecting…";
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    ws.close();
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function requestState() {
  send({ command: "get_state" });
}

function handleMessage(msg) {
  if (msg.type === "state") {
    const d = msg.data;
    worldDims = d.dimensions || worldDims;
    const step = d.step_count || 0;
    const sInc = (d.s_functional && d.s_functional.s_increment) || 0;
    infoEl.textContent = `Step ${step}  |  S=${sInc.toFixed(6)}  |  agents=${(d.agent_positions || []).length}  |  chunks=${d.active_chunks || "?"}`;
    pushChartPoint(step, sInc);

    // After getting state, request chunk 0,0,0 to render (simple demo)
    if (d.chunk_grid_shape) {
      for (let cx = 0; cx < d.chunk_grid_shape[0]; cx++) {
        for (let cy = 0; cy < d.chunk_grid_shape[1]; cy++) {
          for (let cz = 0; cz < d.chunk_grid_shape[2]; cz++) {
            send({ command: "get_chunk", x: cx, y: cy, z: cz });
          }
        }
      }
    }
  } else if (msg.type === "chunk") {
    renderChunk(msg);
  } else if (msg.type === "chunk_updated") {
    send({ command: "get_chunk", x: msg.x, y: msg.y, z: msg.z });
  }
}

function renderChunk(msg) {
  // For now, clear and rebuild (future: diff-based updates).
  scene.remove(voxelGroup);
  voxelGroup = new THREE.Group();

  const shape = msg.shape; // [sx, sy, sz]
  if (!msg.data_b64) return;

  const raw = Uint8Array.from(atob(msg.data_b64), (c) => c.charCodeAt(0));
  // Skip 12-byte header (3 x uint32)
  const floats = new Float32Array(raw.buffer, 12);

  const thresholds = [0.15, 0.30, 0.60, 0.80]; // void, air, soil, bedrock

  let idx = 0;
  const ox = msg.x * (shape[0]);
  const oy = msg.y * (shape[1]);
  const oz = msg.z * (shape[2]);

  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < shape[1]; j++) {
      for (let k = 0; k < shape[2]; k++) {
        const val = floats[idx++];
        let band = 0;
        if (val >= thresholds[3]) band = 4;
        else if (val >= thresholds[2]) band = 3;
        else if (val >= thresholds[1]) band = 2;
        else if (val >= thresholds[0]) band = 1;

        if (band === 0) continue; // skip void

        const mat = materials[band];
        if (!mat) continue;
        const mesh = new THREE.Mesh(boxGeo, mat);
        mesh.position.set(ox + i, oy + j, oz + k);
        voxelGroup.add(mesh);
      }
    }
  }

  scene.add(voxelGroup);
}

// ---------- Render Loop ----------
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ---------- Polling ----------
setInterval(() => {
  if (!paused) requestState();
}, POLL_INTERVAL_MS);

// Handle resize
window.addEventListener("resize", () => {
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
});

connect();
