/* 小智客服聊天页逻辑:fetch + ReadableStream 消费 POST SSE(EventSource 不支持 POST)。
 * ch06 增量:
 *   - 订单选择器:缺订单号的退款请求,后端回 order_options,前端渲染可点选的订单卡片,
 *     点一张就用「同一句原话 + 这个订单号」重发,流程接着走;
 *   - 退款单:Agent 判断完给「申请退款」按钮,点开是固定类目下拉的简单表单。
 * ch05 增量:
 *   - 顶部流程行:把图走过的节点按顺序显示出来(验收要求"看得见强制检索被走到");
 *   - 「转人工」「建工单」两个按钮:互不绑定,点哪个算哪个,都不点就继续聊。
 * ch04 增量:
 *   - citations 事件:把答案里的 [n] 渲染成可点角标,点开右侧面板看来源章节与原文;
 *   - 每条回答挂 👍/👎,点一次点亮并锁定(纯前端采集,为后面的数据飞轮留入口)。
 */

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");
const sourcePanel = document.getElementById("source-panel");
const sourceBody = document.getElementById("source-body");
const sourceClose = document.getElementById("source-close");
const convListEl = document.getElementById("conv-list");

// 会话标识:session_id 是用户键(同一浏览器连续聊);conversation_id 由服务端
// done 事件回传。新会话按钮置 pendingNewConversation——下一条消息带
// new_conversation=true 让服务端真新建(服务端 None 的语义是复用)。
// `?user=xxx` 可覆盖用户键:换条会话看历史、复现某个 user_key 的问题时不用清缓存。
const userOverride = new URLSearchParams(location.search).get("user");
let sessionId = userOverride || localStorage.getItem("kefu_session_id");
if (!sessionId) {
  sessionId = "web-" + (crypto.randomUUID ? crypto.randomUUID() : Date.now());
  localStorage.setItem("kefu_session_id", sessionId);
}
let conversationId = null;
let pendingNewConversation = false;
// ch06: 上一句原话(点订单卡片时用它重发) 与 已确定的订单号(申请退款要用)
let lastUserMessage = "";
let selectedOrderId = null;

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addUserBubble(text) {
  const msg = document.createElement("div");
  msg.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function addAssistantTurn() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  const badges = document.createElement("div");
  badges.className = "tool-badges";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const sources = document.createElement("div");
  sources.className = "source-list";
  const feedback = document.createElement("div");
  feedback.className = "feedback";
  const actions = document.createElement("div");
  actions.className = "actions";
  msg.appendChild(badges);
  msg.appendChild(bubble);
  msg.appendChild(sources);
  msg.appendChild(actions);
  msg.appendChild(feedback);
  messagesEl.appendChild(msg);
  scrollToBottom();
  // 徽章按 tool_call_id 对号:一轮可能并行调多个工具,事件流是
  // start(A) start(B) end(A) end(B),必须按 id 更新。
  return {
    msg,
    badges,
    bubble,
    sources,
    actions,
    feedback,
    badgeMap: new Map(),
    citations: [],
    flow: null,
    flowNodes: [],
  };
}

function addBadge(turn, { id, label, cls }) {
  const badge = document.createElement("span");
  badge.className = "badge " + cls;
  badge.textContent = label;
  turn.badges.appendChild(badge);
  if (id) turn.badgeMap.set(id, badge);
  return badge;
}

/* ---------------------------------------------------------------- 流程行 */

// 节点名 → 人话。顺序就是图里真实经过的顺序(事件流本身有序,不再排序)。
const NODE_LABELS = {
  resolve_reference: "指代消解",
  classify_intent: "意图识别",
  retrieve_knowledge: "知识检索",
  confidence_gate: "置信度闸",
  agent: "主力Agent",
  abstain: "兜底话术",
  complaint_reply: "投诉安抚",
  chitchat_reply: "闲聊话术",
  log_trace: "日志",
};

function appendFlowNode(turn, name) {
  const label = NODE_LABELS[name] || name;
  if (turn.flowNodes[turn.flowNodes.length - 1] === label) return; // 进出各一次,只记一次
  turn.flowNodes.push(label);

  if (!turn.flow) {
    turn.flow = document.createElement("div");
    turn.flow.className = "flow-line";
    turn.msg.insertBefore(turn.flow, turn.bubble);
  }
  if (turn.flow.childElementCount) {
    const arrow = document.createElement("span");
    arrow.className = "flow-arrow";
    arrow.textContent = "→";
    turn.flow.appendChild(arrow);
  }
  const chip = document.createElement("span");
  chip.className = "flow-node" + (name === "retrieve_knowledge" ? " hit" : "");
  chip.textContent = label;
  turn.flow.appendChild(chip);
}

function markIntent(turn, data) {
  if (!turn.flow) {
    turn.flow = document.createElement("div");
    turn.flow.className = "flow-line";
    turn.msg.insertBefore(turn.flow, turn.bubble);
  }
  const chip = document.createElement("span");
  chip.className = "flow-node flow-intent";
  chip.textContent = "意图:" + data.intent;
  turn.flow.appendChild(chip);
}

/* ---------------------------------------------------------------- 动作按钮 */

// 两个按钮各管各的:点了一个不影响另一个,都不点就继续正常对话。
function renderActions(turn, actions) {
  if (!actions || !actions.length) return;
  turn.hasActions = true;
  actions.forEach((action) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "action-btn";
    btn.textContent = action.label;
    btn.dataset.action = action.id;
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.classList.add("done");
      btn.textContent = action.label + "(已处理)";
      if (action.id === "transfer_human") await doTransfer(btn);
      else if (action.id === "create_ticket") await doCreateTicket(btn);
      else if (action.id === "apply_refund") await openRefundForm(selectedOrderId);
    });
    turn.actions.appendChild(btn);
  });
}

function addSystemBubble(text, cls) {
  const msg = document.createElement("div");
  msg.className = "msg " + (cls || "system");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function addAssistantBubble(text) {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
}

// 转人工:本章是前端模拟,不接真人系统——只展示"已转接"+ 客服小猫问候。
async function doTransfer(btn) {
  try {
    const resp = await fetch("/api/v1/chat/actions/transfer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    addSystemBubble(data.notice);
    addAssistantBubble(data.greeting);
  } catch (err) {
    btn.disabled = false;
    btn.classList.remove("done");
    btn.textContent = "转人工";
    showError(String(err));
  }
}

// 建工单:这一步才真的写库(后端调第 2 章的工单工具)。
async function doCreateTicket(btn) {
  try {
    const resp = await fetch("/api/v1/chat/actions/create-ticket", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
    addSystemBubble("工单已创建:" + (data.ticket_no || "已受理"));
    addAssistantBubble(data.message);
  } catch (err) {
    btn.disabled = false;
    btn.classList.remove("done");
    btn.textContent = "建工单";
    showError(String(err));
  }
}

/* ---------------------------------------------------------------- 订单选择器 */

function renderOrderCards(turn, orders) {
  if (!orders || !orders.length) return;
  const wrap = document.createElement("div");
  wrap.className = "order-cards";
  const hint = document.createElement("div");
  hint.className = "order-hint";
  hint.textContent = "请点选一笔订单:";
  wrap.appendChild(hint);

  orders.forEach((o) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "order-card";
    card.dataset.orderId = o.order_id;
    const top = document.createElement("div");
    top.className = "order-top";
    top.textContent = "订单 " + o.order_id + " · " + o.status;
    const mid = document.createElement("div");
    mid.className = "order-mid";
    mid.textContent = o.product + " × " + o.quantity;
    const bottom = document.createElement("div");
    bottom.className = "order-bottom";
    bottom.textContent = "实付 ¥" + Number(o.amount).toFixed(2) + " · 下单 " + o.order_date;
    card.append(top, mid, bottom);
    card.addEventListener("click", () => pickOrder(wrap, o.order_id));
    wrap.appendChild(card);
  });
  turn.msg.appendChild(wrap);
  scrollToBottom();
}

// 点选后锁定整组卡片,用同一句原话 + 选中订单号重发,让子流程从断点继续
function pickOrder(wrap, orderId) {
  wrap.querySelectorAll(".order-card").forEach((c) => {
    c.disabled = true;
    if (c.dataset.orderId === orderId) c.classList.add("picked");
  });
  selectedOrderId = orderId;
  addSystemBubble("已选择订单 " + orderId);
  if (lastUserMessage) sendMessage(lastUserMessage, orderId);
}

/* ---------------------------------------------------------------- 工单确认卡片 */
// CH08:写操作(建工单)后端挂起等人点按钮。卡片显示的就是后端要提交的内容,
// 点「确认提交」时把这两项原样带回去——后端会比对,对不上按拒绝处理。

let pendingConfirm = null;

function renderConfirmCard(turn, data) {
  const wrap = document.createElement("div");
  wrap.className = "confirm-card";
  wrap.dataset.state = "pending";

  const title = document.createElement("div");
  title.className = "confirm-title";
  title.textContent = "提交工单 · " + (data.ticket_type || "咨询");
  const desc = document.createElement("div");
  desc.className = "confirm-desc";
  desc.textContent = data.description || "(无描述)";
  const note = document.createElement("div");
  note.className = "confirm-note";
  note.textContent = "确认后才会提交给人工客服,取消不建单。";

  const row = document.createElement("div");
  row.className = "confirm-actions";
  const ok = document.createElement("button");
  ok.type = "button";
  ok.className = "confirm-btn primary";
  ok.textContent = "确认提交";
  const no = document.createElement("button");
  no.type = "button";
  no.className = "confirm-btn";
  no.textContent = "取消";
  row.append(ok, no);

  wrap.append(title, desc, note, row);
  turn.msg.appendChild(wrap);
  scrollToBottom();

  const payload = { ticket_type: data.ticket_type || "", description: data.description || "" };
  ok.addEventListener("click", () => submitConfirm(wrap, turn, "approve", payload));
  no.addEventListener("click", () => submitConfirm(wrap, turn, "reject", payload));
  pendingConfirm = { turn, wrap, payload };
}

async function submitConfirm(wrap, turn, decision, payload) {
  if (wrap.dataset.state !== "pending") return;
  wrap.dataset.state = decision;
  wrap.querySelectorAll("button").forEach((b) => {
    b.disabled = true;
  });
  const tag = document.createElement("div");
  tag.className = "confirm-note";
  tag.textContent = decision === "approve" ? "已确认,正在提交…" : "已取消,不提交工单。";
  wrap.appendChild(tag);
  pendingConfirm = null;

  sendBtn.disabled = true;
  try {
    const resp = await fetch("/api/v1/chat/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        conversation_id: conversationId,
        decision,
        ticket_type: payload.ticket_type,
        description: payload.description,
      }),
    });
    if (!resp.ok || !resp.body) {
      showError("HTTP " + resp.status);
      return;
    }
    await consumeSSE(resp, turn);
  } catch (err) {
    showError(String(err));
  } finally {
    sendBtn.disabled = false;
  }
}

/* ---------------------------------------------------------------- 退款单 */

let refundReasons = null;

async function loadRefundReasons() {
  if (refundReasons) return refundReasons;
  try {
    const resp = await fetch("/api/v1/refund/reasons");
    const data = await resp.json();
    refundReasons = data.reasons || [];
  } catch (err) {
    refundReasons = [];
  }
  return refundReasons;
}

async function openRefundForm(orderId) {
  if (!orderId) {
    showError("还没有确定订单号,请先点选订单");
    return;
  }
  const reasons = await loadRefundReasons();
  const box = document.createElement("div");
  box.className = "refund-form";
  const title = document.createElement("div");
  title.className = "refund-title";
  title.textContent = "申请退款 · 订单 " + orderId;

  const label = document.createElement("label");
  label.textContent = "退款原因";
  const select = document.createElement("select");
  reasons.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    select.appendChild(opt);
  });
  label.appendChild(select);

  const row = document.createElement("div");
  row.className = "refund-actions";
  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "action-btn primary";
  submit.textContent = "提交申请";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "action-btn";
  cancel.textContent = "取消";
  row.append(submit, cancel);
  box.append(title, label, row);
  messagesEl.appendChild(box);
  scrollToBottom();

  cancel.addEventListener("click", () => box.remove());
  submit.addEventListener("click", async () => {
    submit.disabled = true;
    cancel.disabled = true;
    try {
      const resp = await fetch("/api/v1/refund/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          order_id: orderId,
          reason: select.value,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : "HTTP " + resp.status);
      box.remove();
      addSystemBubble("退款申请已提交:" + (data.refund_no || "已受理"));
      addAssistantBubble(data.message);
    } catch (err) {
      submit.disabled = false;
      cancel.disabled = false;
      showError(String(err));
    }
  });
}

function showError(text) {
  const line = document.createElement("div");
  line.className = "error-line";
  line.textContent = "出错了:" + text;
  messagesEl.appendChild(line);
  scrollToBottom();
}

/* ---------------------------------------------------------------- 引用 */

function openSource(citation) {
  if (!citation) return;
  sourceBody.innerHTML = "";
  const where = document.createElement("div");
  where.className = "source-where";
  where.textContent = "章节:" + (citation.section_path || citation.question || "知识库");
  const q = document.createElement("div");
  q.className = "source-q";
  q.textContent = citation.question || "";
  const a = document.createElement("div");
  a.className = "source-a";
  a.textContent = citation.answer || "";
  const meta = document.createElement("div");
  meta.className = "source-meta";
  meta.textContent = "chunk #" + citation.chunk_id + " · 相关度 " + Number(citation.score).toFixed(4);
  sourceBody.append(where, q, a, meta);
  sourcePanel.classList.add("open");
}

// 把答案文本里的 [n] 变成可点角标。用 DOM 拼,不用 innerHTML——答案来自模型,
// 直接塞 HTML 等于给自己开一个注入口子。
function decorateCitations(turn) {
  const text = turn.bubble.textContent;
  const byN = new Map(turn.citations.map((c) => [String(c.n), c]));
  turn.bubble.textContent = "";
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      turn.bubble.appendChild(document.createTextNode(text.slice(last, m.index)));
    }
    const n = m[1];
    const citation = byN.get(n);
    if (!citation) {
      turn.bubble.appendChild(document.createTextNode(m[0])); // 模型编的编号:原样显示
    } else {
      const sup = document.createElement("button");
      sup.className = "cite";
      sup.type = "button";
      sup.textContent = n;
      sup.title = citation.section_path || "查看来源";
      sup.addEventListener("click", () => openSource(citation));
      turn.bubble.appendChild(sup);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) turn.bubble.appendChild(document.createTextNode(text.slice(last)));
  scrollToBottom();
}

function renderSources(turn) {
  turn.sources.innerHTML = "";
  if (!turn.citations.length) return;
  turn.citations.forEach((c) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "source-chip";
    item.textContent = "[" + c.n + "] " + (c.section_path || c.question || "知识库");
    item.addEventListener("click", () => openSource(c));
    turn.sources.appendChild(item);
  });
}

/* ---------------------------------------------------------------- 反馈 */

// CH09：👎 要接后端（落进低置信度问题池），所以把「当轮原话 + 当轮召回结果」
// 一起带上去——后端拿它存召回快照，审核人靠快照判断是知识缺了还是没检到。
function sendFeedback(turn, rating) {
  const payload = {
    conversation_id: turn.conversationId || conversationId || null,
    question: turn.question || lastUserMessage || "",
    rating,
    citations: (turn.citations || []).map((c) => ({
      n: c.n,
      chunk_id: c.chunk_id,
      section_path: c.section_path,
      score: c.score,
      snippet: c.answer,
    })),
  };
  if (!payload.question) return; // 没有原话就没法入池，也没法统计
  fetch("/api/v1/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {
    // 反馈失败不打扰用户：他点的是「有没有帮助」，不是「提交表单」
  });
}

function renderFeedback(turn) {
  if (turn.feedback.childElementCount) return; // 已渲染过
  const wrap = document.createElement("div");
  wrap.className = "feedback-inner";
  const label = document.createElement("span");
  label.className = "feedback-label";
  label.textContent = "这个回答有帮助吗?";
  const up = document.createElement("button");
  up.type = "button";
  up.className = "fb-btn";
  up.textContent = "👍";
  const down = document.createElement("button");
  down.type = "button";
  down.className = "fb-btn";
  down.textContent = "👎";
  const done = document.createElement("span");
  done.className = "feedback-done";
  done.textContent = "已反馈";
  done.hidden = true;

  function pick(btn, other, value) {
    btn.classList.add("active");
    other.classList.add("locked");
    btn.disabled = true;
    other.disabled = true;
    wrap.dataset.rating = value;
    done.hidden = false; // 一次性锁定:点过就不能再改
    sendFeedback(turn, value);
  }
  up.addEventListener("click", () => pick(up, down, "up"));
  down.addEventListener("click", () => pick(down, up, "down"));
  wrap.append(label, up, down, done);
  turn.feedback.appendChild(wrap);
}

/* ---------------------------------------------------------------- 事件 */

function handleEvent(event, data, turn) {
  switch (event) {
    case "node":
      if (data.phase === "exit") appendFlowNode(turn, data.node);
      break;
    case "intent":
      markIntent(turn, data);
      break;
    case "resolved":
      if (data.order_id) selectedOrderId = data.order_id;
      if (data.changed) addSystemBubble("已补全为:" + data.resolved);
      break;
    case "order_options":
      renderOrderCards(turn, data.orders);
      break;
    case "gate":
      if (!data.passed) {
        addSystemBubble("证据不足,已转入兜底话术(可直接转人工或建工单)");
      }
      break;
    case "tool_start":
      addBadge(turn, {
        id: data.tool_call_id,
        label: "⏳ " + data.name + " 调用中…",
        cls: "running",
      });
      break;
    case "tool_end": {
      const badge = turn.badgeMap.get(data.tool_call_id);
      if (badge) {
        badge.textContent = (data.ok ? "✅ " : "❌ ") + data.name;
        badge.className = "badge " + (data.ok ? "ok" : "fail");
      }
      break;
    }
    case "token":
      turn.bubble.textContent += data.content;
      scrollToBottom();
      break;
    case "citations":
      turn.citations = data.citations || [];
      decorateCitations(turn);
      renderSources(turn);
      break;
    case "interrupt":
      renderConfirmCard(turn, data);
      break;
    case "done":
      conversationId = data.conversation_id;
      turn.conversationId = data.conversation_id;
      loadConversations(); // 新会话会出现在侧栏
      if (data.finish_reason === "abstain") turn.bubble.classList.add("abstained");
      renderActions(turn, data.actions);
      // 挂在确认上的那一轮还没答完,先别问「有帮助吗」
      if (!data.interrupted) renderFeedback(turn);
      break;
    case "error":
      showError(data.message || data.code);
      break;
    default:
      break;
  }
}

/* ---------------------------------------------------------------- 多会话侧栏 */

// 侧栏加载失败只降级,不影响聊天(需求 7)
async function loadConversations() {
  if (!convListEl) return;
  try {
    const resp = await fetch("/api/v1/conversations?user_key=" + encodeURIComponent(sessionId));
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    renderConversations(data.conversations || []);
  } catch (err) {
    convListEl.innerHTML = "";
  }
}

function renderConversations(items) {
  convListEl.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "conv-item";
    empty.textContent = "还没有会话,先聊一句吧";
    convListEl.appendChild(empty);
    return;
  }
  items.forEach((c) => {
    const item = document.createElement("div");
    item.className = "conv-item" + (c.conversation_id === conversationId ? " active" : "");
    item.dataset.convId = c.conversation_id;
    const text = document.createElement("div");
    text.textContent = c.preview || "(空会话)";
    if (c.summarized) {
      const tag = document.createElement("span");
      tag.className = "conv-tag";
      tag.textContent = "已摘要";
      text.appendChild(tag);
    }
    const meta = document.createElement("div");
    meta.className = "conv-meta";
    meta.textContent = c.message_count + " 条 · " + (c.created_at ? c.created_at.slice(5, 16).replace("T", " ") : "");
    item.append(text, meta);
    item.addEventListener("click", () => openConversation(c.conversation_id));
    convListEl.appendChild(item);
  });
}

// 切回旧会话:回载历史原文(接口给的是原文,不是梗概)
async function openConversation(id) {
  if (!id) return;
  try {
    const resp = await fetch(
      "/api/v1/conversations/" + id + "/messages?user_key=" + encodeURIComponent(sessionId)
    );
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    conversationId = id;
    pendingNewConversation = false;
    selectedOrderId = null;
    messagesEl.innerHTML = "";
    (data.messages || []).forEach((m) => {
      if (m.role === "user") addUserBubble(m.content);
      else if (m.role === "assistant" && m.content) addAssistantBubble(m.content);
    });
    if (data.summarized) addSystemBubble("这段早期对话已被压缩成梗概(原文仍在上方)");
    loadConversations();
    inputEl.focus();
  } catch (err) {
    showError("会话历史加载失败:" + err);
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  sendMessage(text, null);
}

// 读 SSE 帧并派发给 handleEvent。首轮问答与「确认工单后接着跑」共用这一段:
// resume 的响应体格式完全一样,区别只在 turn 是同一个(答复接着往上写)。
async function consumeSSE(resp, turn) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // 最后半帧留到下轮
    for (const frame of frames) {
      const eventMatch = frame.match(/^event:\s*(.+)$/m);
      const dataMatch = frame.match(/^data:\s*(.+)$/m);
      if (eventMatch && dataMatch) {
        try {
          handleEvent(eventMatch[1].trim(), JSON.parse(dataMatch[1]), turn);
        } catch (err) {
          console.error("bad SSE frame:", frame, err);
        }
      }
    }
  }
}

// orderId 只在「点选订单卡片」这条路上传:普通发消息一律不带,免得一直粘着上一单
async function sendMessage(text, orderId) {
  if (!text) return;
  lastUserMessage = text;
  if (orderId) selectedOrderId = orderId;
  addUserBubble(text);
  sendBtn.disabled = true;
  const turn = addAssistantTurn();
  turn.question = text; // CH09：👎 时要带当轮原话入池

  try {
    const resp = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        conversation_id: pendingNewConversation ? null : conversationId,
        new_conversation: pendingNewConversation,
        order_id: orderId || null,
      }),
    });
    pendingNewConversation = false;
    if (!resp.ok || !resp.body) {
      showError("HTTP " + resp.status);
      return;
    }
    await consumeSSE(resp, turn);
  } catch (err) {
    showError(String(err));
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sourceClose.addEventListener("click", () => sourcePanel.classList.remove("open"));
sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) send();
});
newChatBtn.addEventListener("click", () => {
  pendingNewConversation = true;
  conversationId = null;
  selectedOrderId = null;
  messagesEl.innerHTML = "";
  loadConversations();
});

// CH07:进页面先拉会话列表;失败静默,聊天照常可用
loadConversations();
