from flask import Flask, render_template, request, jsonify
from backend import chatbot_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        uploaded_content = uploaded_file.read().decode('utf-8')
        return 0

if __name__ == '__main__':
    app.run(debug=False)
