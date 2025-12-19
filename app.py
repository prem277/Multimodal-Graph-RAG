import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from query_data import answer_question

app = Flask(__name__)
CORS(app)

# ---------------- Chat history ----------------
USER_SESSIONS = {}

def get_session_history(session_id):
    if session_id not in USER_SESSIONS:
        USER_SESSIONS[session_id] = ""
    return USER_SESSIONS[session_id]

def update_session_history(session_id, user_q, bot_ans):
    USER_SESSIONS[session_id] += f"Question: {user_q}\nAssistant: {bot_ans}\n\n"

# ---------------- Frontend ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Chat</title>
    <style>
        body { font-family: Arial, sans-serif; background:#f5f5f5; padding:20px;}
        #chat-box { width: 100%; height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; background:white; }
        #user-input { width: 80%; padding: 10px; }
        #send-btn { padding: 10px; }
        .user { color: blue; }
        .bot { color: green; }
        .message { margin:5px 0; }
    </style>
</head>
<body>
    <h2>RAG Chat Interface</h2>
    <div id="chat-box"></div>
    <input type="text" id="user-input" placeholder="Ask your question..." />
    <button id="send-btn">Send</button>

    <script>
        const chatBox = document.getElementById('chat-box');
        const input = document.getElementById('user-input');
        const button = document.getElementById('send-btn');
        let session_id = 'default';

        function appendMessage(sender, text){
            const div = document.createElement('div');
            div.className = 'message ' + sender;
            div.innerHTML = '<b>' + sender + ':</b> ' + text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        button.onclick = async () => {
            const q = input.value.trim();
            if(!q) return;
            appendMessage('Question', q);
            input.value = '';

            const resp = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, session_id })
            });
            const data = await resp.json();
            appendMessage('Assistant', data.answer || '[No answer]');
        };

        input.addEventListener("keypress", function(e) {
            if(e.key === 'Enter'){ button.click(); }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

# ---------------- API ----------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    history = get_session_history(session_id)

    try:
        answer = answer_question(question, history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    update_session_history(session_id, question, answer)

    return jsonify({"answer": answer, "session_id": session_id})

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
