// Semantic Search — the retrieval step behind RAG, running fully in the browser.
// A real sentence-embedding model (all-MiniLM-L6-v2) is loaded via Transformers.js;
// the query and a small corpus are embedded locally and ranked by cosine similarity.

const REPO_URL = "https://github.com/tongchen2010/semantic-search-rag";
const LIB_URL = "https://esm.run/@huggingface/transformers";
const MODEL = "Xenova/all-MiniLM-L6-v2";

const CORPUS = [
  "The cat slept on the warm windowsill all afternoon.",
  "A small feline napped in the sunshine by the glass door.",
  "Quantum computers use qubits to perform certain calculations.",
  "The stock market fell sharply after the surprise announcement.",
  "Investors panicked as share prices tumbled across the board.",
  "She baked a loaf of sourdough bread early on Sunday morning.",
  "The recipe calls for two cups of flour, water, salt and yeast.",
  "Photosynthesis converts sunlight into chemical energy in plants.",
  "Leaves capture light and turn carbon dioxide into sugar.",
  "Machine learning models generally improve with more training data.",
  "A neural network gradually learns patterns from labeled examples.",
  "The marathon runner crossed the finish line utterly exhausted.",
];

const $ = (id) => document.getElementById(id);
const query = $("query"), searchBtn = $("search-btn"), results = $("results"),
  statusEl = $("status"), fill = $("progress-fill"), progressWrap = $("progress-wrap");
$("repo-link").href = REPO_URL;

let extractor = null, corpusEmb = [];

init();
async function init() {
  let transformers;
  try {
    transformers = await import(LIB_URL);
  } catch (e) {
    return fail("Couldn't load Transformers.js (are you online?).");
  }
  try {
    extractor = await transformers.pipeline("feature-extraction", MODEL, {
      progress_callback: (r) => {
        if (r.status === "progress" && r.progress != null) {
          fill.style.width = r.progress + "%";
          statusEl.textContent = `Downloading ${r.file || "model"}… ${Math.round(r.progress)}%`;
        }
      },
    });
    fill.style.width = "100%";
    statusEl.textContent = "Embedding the corpus…";
    corpusEmb = await Promise.all(CORPUS.map(embed));
    ready();
  } catch (e) {
    console.error(e);
    fail("Failed to load the model: " + (e?.message || e));
  }
}

async function embed(text) {
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(out.data);
}

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

async function search() {
  const q = query.value.trim();
  if (!q || !extractor) return;
  searchBtn.disabled = true;
  const qv = await embed(q);
  const ranked = CORPUS
    .map((text, i) => ({ text, score: dot(qv, corpusEmb[i]) }))
    .sort((a, b) => b.score - a.score);
  const top = ranked[0].score;
  results.innerHTML = "";
  ranked.forEach((r, i) => {
    const w = Math.max(2, (r.score / (top || 1)) * 100);
    const el = document.createElement("div");
    el.className = "result" + (i === 0 ? " top" : "");
    el.innerHTML = `<div class="text">${esc(r.text)}</div>
      <div class="meter"><div class="bar"><div style="width:${w}%"></div></div>
      <span class="score">${r.score.toFixed(3)}</span></div>`;
    results.appendChild(el);
  });
  searchBtn.disabled = false;
}

function ready() {
  progressWrap.style.display = "none";
  query.disabled = false; searchBtn.disabled = false;
  document.querySelectorAll(".chip").forEach((c) => {
    c.disabled = false;
    c.addEventListener("click", () => { query.value = c.textContent; search(); });
  });
  query.addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
  searchBtn.addEventListener("click", search);
  query.focus();
}

function fail(msg) { statusEl.textContent = msg; statusEl.classList.add("err"); }
function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
