/* 飞轮审核页（CH09）。原生 JS，无构建。
 *
 * 三条约定：
 * 1. 所有来自库里的文本都用 textContent 塞进 DOM，不做字符串拼 HTML——
 *    这些问题原话是用户随手打的，里面完全可能有 < 或引号。
 * 2. 通过/驳回之后重新拉一遍队列，不在本地改状态——状态机在服务端，
 *    前端猜错了就会出现「界面显示已通过、库里还是待审」。
 * 3. 接口失败时把话说清楚（谁挂了、怎么办），不吞错误。
 */

const API = "/api/v1";

const state = {
  currentId: null,
  queue: [],
  // 允许用 ?status=通过 直接落到某个状态（可分享的链接，也方便截图核对）
  status: new URLSearchParams(location.search).get("status") || "待审",
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} → HTTP ${resp.status}`);
  return resp.json();
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

/* ---------------------------------------------------------------- 队列 */

async function loadQueue() {
  const data = await getJSON(`${API}/review/queue?status=${encodeURIComponent(state.status)}&limit=100`);
  state.queue = data.items;

  const counts = data.counts || {};
  $("queue-badge").textContent = `待审 ${counts["待审"] || 0} · 通过 ${counts["通过"] || 0} · 驳回 ${
    counts["驳回"] || 0
  }`;

  const box = $("queue");
  box.replaceChildren();
  if (!data.items.length) {
    box.append(
      el(
        "p",
        "empty",
        state.status === "待审"
          ? data.hint || "队列是空的。先跑一轮飞轮：python scripts/run_flywheel.py --batch 50"
          : "这个状态下还没有记录。"
      )
    );
    return;
  }
  for (const item of data.items) {
    const row = el("div", "queue-item");
    if (item.id === state.currentId) row.classList.add("active");
    row.append(el("div", "q", item.normalized_question));
    const meta = el("div", "meta");
    meta.append(el("span", "count", `×${item.occurrence_count}`));
    meta.append(el("span", null, item.created_at || ""));
    row.append(meta);
    row.addEventListener("click", () => selectReview(item.id));
    box.append(row);
  }

  if (!state.currentId || !data.items.some((i) => i.id === state.currentId)) {
    selectReview(data.items[0].id);
  }
}

/* ---------------------------------------------------------------- 详情 */

function renderSnapshot(container, snapshot) {
  if (!snapshot || !snapshot.length) return;
  const wrap = el("div", "snap");
  for (const s of snapshot) {
    const row = el("div", "snap-row");
    row.append(el("span", "score", Number(s.score || 0).toFixed(3)));
    row.append(el("span", "path", s.section_path || "—"));
    row.append(el("span", "snippet", s.snippet || ""));
    wrap.append(row);
  }
  container.append(wrap);
}

function renderDetail(data) {
  const box = $("detail");
  box.replaceChildren();

  const head = el("div", "detail-head");
  const headLeft = el("div");
  headLeft.append(el("div", "detail-question", data.normalized_question));
  headLeft.append(
    el(
      "div",
      "detail-meta",
      `#${data.id} · 出现 ${data.occurrence_count} 次 · ${data.review_status} · 入队 ${
        data.created_at || "—"
      }`
    )
  );
  head.append(headLeft);
  box.append(head);

  const field = el("div", "field");
  field.append(el("label", null, "核准答案（会写进知识库；改了就重算向量）"));
  const ta = el("textarea");
  ta.id = "answer";
  ta.value = data.approved_answer || data.ai_suggested_answer || "";
  field.append(ta);
  box.append(field);

  const actions = el("div", "actions");
  const approved = data.review_status === "通过";
  const approve = el("button", "primary", approved ? "更新答案并重写知识库" : "通过并写入知识库");
  const reject = el("button", "danger", data.review_status === "驳回" ? "保持驳回" : "驳回");
  const msg = el("span", "msg");
  approve.addEventListener("click", () => submit(approve, "approve", ta.value, msg));
  reject.addEventListener("click", () => submit(reject, "reject", undefined, msg));
  actions.append(approve, reject, msg);
  box.append(actions);

  const sourcesBox = el("div", "field");
  sourcesBox.append(el("label", null, `贡献这条缺口的原话 ${data.sources.length} 条（含当时的召回快照）`));
  const list = el("div", "sources");
  if (!data.sources.length) {
    list.append(el("p", "empty", "还没有归并进来的原话——这条是飞轮在待审队列里直接建的？"));
  }
  for (const s of data.sources) {
    const card = el("div", "source");
    card.append(el("div", "raw", s.raw_question));
    const tags = el("div", "tags");
    tags.append(el("span", "tag", s.source));
    if (s.reason) tags.append(el("span", "tag gray", s.reason));
    tags.append(el("span", "tag gray", s.created_at || ""));
    card.append(tags);
    renderSnapshot(card, s.snapshot);
    list.append(card);
  }
  sourcesBox.append(list);
  box.append(sourcesBox);
}

async function selectReview(id) {
  state.currentId = id;
  for (const node of document.querySelectorAll(".queue-item")) node.classList.remove("active");
  try {
    const data = await getJSON(`${API}/review/${id}`);
    renderDetail(data);
    // 高亮当前行
    const rows = document.querySelectorAll(".queue-item");
    const idx = state.queue.findIndex((i) => i.id === id);
    if (rows[idx]) rows[idx].classList.add("active");
  } catch (err) {
    $("detail").replaceChildren(el("p", "empty", `详情加载失败：${err.message}`));
  }
}

async function submit(button, action, answer, msgNode) {
  if (!state.currentId) return;
  button.disabled = true;
  msgNode.className = "msg";
  msgNode.textContent = action === "approve" ? "写入中（要算向量，稍等）…" : "处理中…";
  try {
    if (action === "approve") {
      await postJSON(`${API}/review/${state.currentId}/approve`, { answer });
    } else {
      await postJSON(`${API}/review/${state.currentId}/reject`);
    }
    msgNode.className = "msg ok";
    msgNode.textContent = action === "approve" ? "已通过，知识库已更新" : "已驳回";
    state.currentId = null; // 处理过的不再留在待审列表里
    await Promise.all([loadQueue(), loadTrend()]);
  } catch (err) {
    msgNode.className = "msg err";
    msgNode.textContent = `失败：${err.message}`;
    button.disabled = false;
  }
}

/* ---------------------------------------------------------------- 趋势 */

async function loadTrend() {
  const box = $("trend");
  try {
    const data = await getJSON(`${API}/review/trend?limit=12`);
    box.replaceChildren();
    if (!data.items.length) {
      box.append(el("p", "empty", "还没有评估记录。跑一轮：python scripts/run_eval_round.py"));
      return;
    }
    const series = [
      ["R@5", "r5"],
      ["R@10", "r10"],
      ["MRR", "mrr"],
    ];
    for (const [label, key] of series) {
      const row = el("div", "trend-row");
      row.append(el("span", "label", label));
      const track = el("div", "track");
      const values = data.items.map((i) => Number(i[key] || 0));
      const last = values[values.length - 1] || 0;
      const fill = el("div", "fill");
      fill.style.width = `${Math.max(2, Math.min(100, last * 100))}%`;
      track.append(fill);
      row.append(track);
      row.append(el("span", "num", last.toFixed(3)));
      box.append(row);
    }
    const meta = data.items[data.items.length - 1];
    box.append(
      el(
        "p",
        "empty",
        `最近一轮：#${meta.id} · ${meta.triggered_by} · ${meta.dataset_size} 题 · ${meta.strategy || "—"} · ${
          meta.created_at || ""
        }${meta.with_generation ? " · 含生成段" : ""}`
      )
    );
  } catch (err) {
    box.replaceChildren(el("p", "empty", `趋势加载失败：${err.message}`));
  }
}

/* ---------------------------------------------------------------- 按意图 token */

async function loadCards() {
  const box = $("cards");
  const note = $("cards-note");
  try {
    const data = await getJSON(`${API}/usage/by-intent`);
    box.replaceChildren();
    // 按意图 token 占比排前四张
    for (const intent of data.items.slice(0, 4)) {
      const card = el("div", "card");
      card.append(el("b", null, intent.total_tokens.toLocaleString()));
      card.append(el("span", null, `${intent.intent} · ${intent.turns} 轮 · 占 ${intent.share}%`));
      const bar = el("div", "bar");
      const inner = document.createElement("i");
      inner.style.width = `${Math.max(2, Math.min(100, intent.share))}%`;
      bar.append(inner);
      card.append(bar);
      box.append(card);
    }
    if (!data.items.length) {
      box.append(el("p", "empty", "还没有对话成本数据——聊几轮就有了。"));
    }
    note.textContent = data.items.length ? data.note : "";
  } catch (err) {
    note.textContent = `成本统计加载失败：${err.message}`;
  }
}

/* ---------------------------------------------------------------- 启动 */

async function refreshAll() {
  try {
    await Promise.all([loadQueue(), loadTrend(), loadCards()]);
  } catch (err) {
    $("queue").replaceChildren(el("p", "empty", `队列加载失败：${err.message}`));
  }
}

$("refresh").addEventListener("click", refreshAll);
$("tabs").addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (!tab) return;
  state.status = tab.dataset.status;
  state.currentId = null;
  for (const node of document.querySelectorAll(".tab")) node.classList.toggle("active", node === tab);
  loadQueue();
});
// 首次渲染时把 tab 高亮对齐到 URL 里的状态
for (const node of document.querySelectorAll(".tab")) {
  node.classList.toggle("active", node.dataset.status === state.status);
}
refreshAll();
