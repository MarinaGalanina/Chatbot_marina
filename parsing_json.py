import os
import json

data_root='C:/Users/admin/Downloads/chatbot'

with open('C:/Users/admin/Downloads/chatbot/training.txt', 'r') as file:
    input_text = file.read()
pairs = input_text.strip().split('\n\n')

intents = []
for pair in pairs:
    question, answer = pair.split('\n', 1)
    intent = {
        "tag": question.strip(' ?'),
        "patterns": [question.strip(' ?')],
        "responses": [answer.strip()]
    }
    intents.append(intent)


json_data = {
    "intents": intents
}

json_file_path = os.path.join(data_root, 'intents.json')
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)
print(f"JSON data saved to: {json_file_path}")


data_file = open(data_root + '/intents.json').read()
data = json.loads(data_file)