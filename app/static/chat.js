/* 小智客服聊天页逻辑:fetch + ReadableStream 消费 POST SSE(EventSource 不支持 POST)。 */

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");

// 会话标识:session_id 是用户键(同一浏览器连续聊);conversation_id 由服务端
// done 事件回传,新会话按钮清空它,让下一条消息开新的一通对话。
let sessionId = localStorage.getItem("kefu_session_id");
if (!sessionId) {
  sessionId = "web-" + (crypto.randomUUID ? crypto.randomUUID() : Date.now());
  localStorage.setItem("kefu_session_id", sessionId);
}
let conversationId = null;

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
  return { badges, bubble };
}

function addBadge(badges, { label, cls }) {
  const badge = document.createElement("span");
  badge.className = "badge " + cls;
  badge.textContent = label;
  badges.appendChild(badge);
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
      turn.activeBadge = addBadge(turn.badges, {
        label: "⏳ " + data.name + " 调用中…",
        cls: "running",
      });
      break;
    case "tool_end":
      if (turn.activeBadge) {
        turn.activeBadge.textContent = (data.ok ? "✅ " : "❌ ") + data.name;
        turn.activeBadge.className = "badge " + (data.ok ? "ok" : "fail");
        turn.activeBadge = null;
      }
      break;
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
  turn.activeBadge = null;

  try {
    const resp = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        conversation_id: conversationId,
      }),
    });
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
  conversationId = null; // 下一条消息开新会话
  messagesEl.innerHTML = "";
});
