/* 小智客服聊天页逻辑:fetch + ReadableStream 消费 POST SSE(EventSource 不支持 POST)。
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

// 会话标识:session_id 是用户键(同一浏览器连续聊);conversation_id 由服务端
// done 事件回传。新会话按钮置 pendingNewConversation——下一条消息带
// new_conversation=true 让服务端真新建(服务端 None 的语义是复用)。
let sessionId = localStorage.getItem("kefu_session_id");
if (!sessionId) {
  sessionId = "web-" + (crypto.randomUUID ? crypto.randomUUID() : Date.now());
  localStorage.setItem("kefu_session_id", sessionId);
}
let conversationId = null;
let pendingNewConversation = false;

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
  msg.appendChild(badges);
  msg.appendChild(bubble);
  msg.appendChild(sources);
  msg.appendChild(feedback);
  messagesEl.appendChild(msg);
  scrollToBottom();
  // 徽章按 tool_call_id 对号:一轮可能并行调多个工具,事件流是
  // start(A) start(B) end(A) end(B),必须按 id 更新。
  return { badges, bubble, sources, feedback, badgeMap: new Map(), citations: [] };
}

function addBadge(turn, { id, label, cls }) {
  const badge = document.createElement("span");
  badge.className = "badge " + cls;
  badge.textContent = label;
  turn.badges.appendChild(badge);
  if (id) turn.badgeMap.set(id, badge);
  return badge;
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
  }
  up.addEventListener("click", () => pick(up, down, "up"));
  down.addEventListener("click", () => pick(down, up, "down"));
  wrap.append(label, up, down, done);
  turn.feedback.appendChild(wrap);
}

/* ---------------------------------------------------------------- 事件 */

function handleEvent(event, data, turn) {
  switch (event) {
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
    case "done":
      conversationId = data.conversation_id;
      if (data.finish_reason === "abstain") turn.bubble.classList.add("abstained");
      renderFeedback(turn);
      break;
    case "error":
      showError(data.message || data.code);
      break;
    default:
      break;
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  addUserBubble(text);
  inputEl.value = "";
  sendBtn.disabled = true;
  const turn = addAssistantTurn();

  try {
    const resp = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        conversation_id: pendingNewConversation ? null : conversationId,
        new_conversation: pendingNewConversation,
      }),
    });
    pendingNewConversation = false;
    if (!resp.ok || !resp.body) {
      showError("HTTP " + resp.status);
      return;
    }

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
  messagesEl.innerHTML = "";
});
