from flask import Flask, render_template, request
from NeuralLanguageContext.neurallanguagecontext import NeuralLanguageContext

app = Flask(__name__)
context = NeuralLanguageContext()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if context.is_model_trained():
        return context.chatbot_response(userText)
    return 'Put your model!'


@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    uploaded_file =  request.files.get("file")

    if uploaded_file:
        context.create_dataset(uploaded_file)
        context.train_model()

    return "0"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
