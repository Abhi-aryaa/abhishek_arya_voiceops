"""Interactive Web UI for ZentroMart VoiceOps 2026 (plays & records audio + text chat)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from .agent import ZentroMartAgent

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ZentroMart VoiceOps 2026 — AI Support Agent</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --text: #f8fafc; --muted: #94a3b8; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }
    .container { max-width: 960px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 20px; }
    h1 { margin: 0; font-size: 1.4rem; color: var(--accent); }
    .badge { background: #0284c7; color: #fff; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
    .chat-box { background: var(--card); border-radius: 12px; padding: 16px; height: 480px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
    .msg { padding: 10px 14px; border-radius: 10px; max-width: 85%; line-height: 1.45; font-size: 0.95rem; }
    .msg.user { background: #0369a1; align-self: flex-end; }
    .msg.agent { background: #334155; align-self: flex-start; }
    .controls { display: flex; gap: 10px; margin-top: 14px; }
    input[type="text"] { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: #fff; }
    button { padding: 10px 16px; border-radius: 8px; border: none; background: var(--accent); color: #0f172a; font-weight: 600; cursor: pointer; }
    button.secondary { background: #475569; color: #fff; }
    .panel { background: var(--card); border-radius: 12px; padding: 16px; font-size: 0.85rem; }
    .panel h3 { margin-top: 0; color: var(--accent); font-size: 0.95rem; }
    .tag { display: inline-block; background: #0f172a; border: 1px solid #38bdf8; color: #38bdf8; padding: 2px 8px; border-radius: 6px; margin: 2px; font-family: monospace; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>ZentroMart VoiceOps 2026 Support Agent</h1>
        <small style="color: var(--muted)">RAG-Grounded · Identity-Verified Tools · Gnani STT/TTS (timbre-v2.5 / Kaveri)</small>
      </div>
      <span class="badge">Reference Date: 20 Oct 2026</span>
    </header>
    <div class="grid">
      <div>
        <div class="chat-box" id="chat">
          <div class="msg agent">Hello! I am ZentroMart's AI support assistant. Ask a policy question or share your 8-digit order ID to get started.</div>
        </div>
        <div class="controls">
          <input type="text" id="msgInput" placeholder="Type in English, Hindi, or Hinglish..." onkeydown="if(event.key==='Enter') sendText()">
          <button onclick="sendText()">Send</button>
          <button class="secondary" onclick="resetChat()">Reset</button>
        </div>
      </div>
      <div class="panel">
        <h3>Retrieved KB &amp; Catalog Chunks</h3>
        <div id="chunks"><span style="color:var(--muted)">None yet</span></div>
        <h3 style="margin-top:18px;">Tool Calls Executed</h3>
        <div id="tools"><span style="color:var(--muted)">None yet</span></div>
        <h3 style="margin-top:18px;">Decision &amp; Language</h3>
        <div id="decision"><span style="color:var(--muted)">—</span></div>
      </div>
    </div>
  </div>
  <script>
    async function sendText() {
      const inp = document.getElementById('msgInput');
      const text = inp.value.trim();
      if (!text) return;
      inp.value = '';
      appendMsg('user', text);
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: text})
      });
      const data = await res.json();
      appendMsg('agent', data.reply);
      document.getElementById('chunks').innerHTML = (data.retrieved_chunks || []).map(c => `<span class="tag">${c}</span>`).join('') || 'None';
      document.getElementById('tools').innerHTML = (data.tool_calls || []).map(t => `<span class="tag">${t}</span>`).join('') || 'None';
      document.getElementById('decision').innerText = `${data.decision} (${data.language})`;
    }
    async function resetChat() {
      await fetch('/api/reset', {method: 'POST'});
      document.getElementById('chat').innerHTML = '<div class="msg agent">Conversation reset. Hello! I am ZentroMart\\'s AI support assistant.</div>';
    }
    function appendMsg(role, text) {
      const box = document.getElementById('chat');
      const d = document.createElement('div');
      d.className = 'msg ' + role;
      d.innerText = text;
      box.appendChild(d);
      box.scrollTop = box.scrollHeight;
    }
  </script>
</body>
</html>
"""


def run_web_server(dataset_dir: Optional[Path] = None, port: int = 8000) -> None:
    agent = ZentroMartAgent(dataset_dir=dataset_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            if self.path == "/api/reset":
                agent.reset()
                out = {"status": "ok"}
            else:
                payload = json.loads(raw)
                msg = payload.get("message", "")
                out = agent.respond(msg, is_voice=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))

    print(f"Starting ZentroMart VoiceOps Web UI on http://127.0.0.1:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
