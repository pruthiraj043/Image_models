from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

messages = []  # Store chat messages

@app.route("/send", methods=["POST"])
def send_message():
    data = request.json
    user = data.get("user")
    message = data.get("message")
    if not user or not message:
        return jsonify({"error": "User and message required"}), 400
    messages.append({"user": user, "message": message})
    return jsonify({"status": "Message sent"})

@app.route("/receive", methods=["GET"])
def receive_messages():
    return jsonify(messages)

def run_server():
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.start()


#curl -X POST http://127.0.0.1:5000/send -H "Content-Type: application/json" -d '{"user": "Alice", "message": "Hello, world!"}'

# import requests

# response = requests.get("http://127.0.0.1:5000/receive")
# print(response.json())  # Output: [{"user": "Alice", "message": "Hello, world!"}]
