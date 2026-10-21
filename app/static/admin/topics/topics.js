/* 主题分布页（CH10）。只读：所有数据来自 /api/v1/topics/*，
   页面上没有任何「跑归类」的按钮——那是脚本/定时任务的事。 */

const API = "/api/v1/topics";
const state = { activeLabel: null };

const el = (id) => document.getElementById(id);

function fmtTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getMonth() + 1}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json();
}

function renderStats(data) {
  const box = el("stats");
  const rows = [
    ["已归类问题", data.total],
    ["池内问题", data.pool_total],
    ["去重后", data.distinct],
    ["命中标签", data.labels.reduce((sum, item) => sum + item.count, 0)],
  ];
  box.innerHTML = rows
    .map(([name, value]) => `<div class="stat"><b>${value}</b><span>${name}</span></div>`)
    .join("");

  const hint = el("hint");
  if (data.empty) {
    hint.textContent = data.hint || "还没有归类结果";
  } else if (data.pool_total > data.distinct) {
    // 池子里同一条问题会因为多次测试重复入池，只报 39 会高估样本量
    hint.textContent =
      `注意：池内 ${data.pool_total} 行只对应 ${data.distinct} 个不同问题（重复测试入池），` +
      "分布按行统计。";
  } else {
    hint.textContent = "";
  }
}

function renderBadge(model) {
  const badge = el("model-badge");
  if (!model.enabled) {
    badge.textContent = "旁路未开启";
    badge.className = "badge";
    badge.title = "TOPIC_ENABLED=false，归类任务跑起来会直接跳过";
  } else if (model.available) {
    badge.textContent = `模型就绪 · 阈值 ${model.threshold}`;
    badge.className = "badge ok";
    badge.title = "本进程已加载分类器";
  } else if (model.error) {
    badge.textContent = "模型不可用";
    badge.className = "badge bad";
    badge.title = model.error;
  } else {
    // 加载失败才标红。只是本进程还没跑过归类，标红会吓人一跳
    badge.textContent = "本进程未加载模型";
    badge.className = "badge";
    badge.title = "分类器按需加载：跑一次归类脚本或重启服务后才会有状态";
  }
}

function renderBars(labels) {
  const box = el("bars");
  const max = Math.max(1, ...labels.map((item) => item.count));
  box.innerHTML = labels
    .map((item) => {
      const width = Math.round((item.count / max) * 100);
      const active = state.activeLabel === item.label ? " active" : "";
      return `<div class="bar-row${active}" data-label="${item.label}">
        <span class="label">${item.label}</span>
        <span class="track"><span class="fill" style="width:${width}%"></span></span>
        <span class="num">${item.count} <small>${item.share}%</small></span>
      </div>`;
    })
    .join("");

  box.querySelectorAll(".bar-row").forEach((row) => {
    row.addEventListener("click", () => {
      const label = row.dataset.label;
      state.activeLabel = state.activeLabel === label ? null : label;
      refreshList();
      renderBars(labels);
    });
  });
}

function renderRows(items) {
  const tbody = el("list").querySelector("tbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td class="empty" colspan="3">没有记录</td></tr>`;
    return;
  }
  tbody.innerHTML = items
    .map(
      (item) => `<tr>
        <td>${item.text}</td>
        <td class="labels">${(item.labels || [])
          .map((label) => `<span class="tag">${label}</span>`)
          .join("")}</td>
        <td class="time">${fmtTime(item.classified_at)}</td>
      </tr>`
    )
    .join("");
}

async function refreshList() {
  const title = el("list-title");
  try {
    if (state.activeLabel) {
      const data = await getJSON(
        `${API}/by-label?label=${encodeURIComponent(state.activeLabel)}&limit=50`
      );
      title.textContent = `类目「${state.activeLabel}」下的问题（${data.count}）`;
      renderRows(data.items);
    } else {
      const data = await getJSON(`${API}/latest?limit=50`);
      title.textContent = `最近归类（${data.count}）`;
      renderRows(data.items);
    }
  } catch (err) {
    title.textContent = "最近归类";
    renderRows([]);
    el("hint").textContent = `列表加载失败：${err.message}`;
  }
}

async function refresh() {
  try {
    const data = await getJSON(`${API}/distribution`);
    renderStats(data);
    renderBadge(data.model);
    renderBars(data.labels);
  } catch (err) {
    el("hint").textContent = `分布加载失败：${err.message}`;
  }
  await refreshList();
}

el("refresh").addEventListener("click", refresh);
refresh();
