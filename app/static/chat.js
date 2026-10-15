/* 小智客服聊天页逻辑:fetch + ReadableStream 消费 POST SSE(EventSource 不支持 POST)。 */

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");

// 会话标识:session_id 是用户键(同一浏览器连续聊);conversation_id 由服务端
// done 事件回传。新会话按钮置 pendingNewConversation——下一条消息带
// new_conversation=true 让服务端真新建(review I2:服务端 None 的语义是复用,
// 前端单方面清 conversation_id 并不能开新会话)。
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
  msg.appendChild(badges);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
  // 徽章按 tool_call_id 对号(review I3):一轮可能并行调多个工具,
  // 事件流是 start(A) start(B) end(A) end(B),必须按 id 更新,
  // 不能用单个 activeBadge 指针(会张冠李戴、丢事件)。
  return { badges, bubble, badgeMap: new Map() };
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
    case "done":
      conversationId = data.conversation_id;
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

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) send();
});
newChatBtn.addEventListener("click", () => {
  // 只清 UI 不够(review I2):置 pendingNewConversation,下一条消息让服务端真新建。
  pendingNewConversation = true;
  messagesEl.innerHTML = "";
});
