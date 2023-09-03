class DataContext:
    def __init__(self):
        self.intents = []
        self.json_data = None

    def _load_file(self, file):
        return file.read().decode("utf-8")

    def preprocessing_file(self, file_path):
        input_text = self._load_file(file_path)
        pairs = input_text.strip().split('\r\n\r\n')
        for pair in pairs:
            question, answer = pair.split('\n', 1)
            intent = {
                "tag": question.strip(' ?'),
                "patterns": [question.strip(' ?')],
                "responses": [answer.strip()]
            }
            self.intents.append(intent)
        self.json_data = {
            "intents": self.intents
        }

    def is_json_data(self):
        return self.json_data is not None

    def get_data(self):
        return self.json_data
