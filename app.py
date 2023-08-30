from flask import Flask, render_template, request, jsonify
import backend
from backend import chatbot_response, train_model
from parsing_json import crete_dataset
import os

app = Flask(__name__)
language_model = None

words_ = None
data_ = None
classes_ = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if language_model is None:
        return "Sorry! I don't understand."
    return chatbot_response(userText, language_model, data_, words_, classes_)

@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    uploaded_file = request.files.get("file")

    if uploaded_file:
        file_path = os.path.join(os.getcwd(), "uploads", uploaded_file.filename)
        uploaded_file.save(file_path)

        # Process the uploaded file (e.g., read contents)
        with open(file_path, "r") as file:
            file_contents = file.read()
            global data_
            intents, pairs, data_ = crete_dataset(file_path)
            global words_, classes_, language_model
            language_model, words_, classes_ = train_model(data_)
    # return 0
        return f"Uploaded and processed file with {len(file_contents)} characters."
    else:
        return "No file uploaded."


@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        uploaded_content = uploaded_file.read().decode('utf-8')
        return 0

if __name__ == '__main__':
    app.run(debug=False)
