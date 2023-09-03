import random
from DataContext.data_context import DataContext
from LanguageModel.language_model import ModelContext


class NeuralLanguageContext:
    def __init__(self):
        self.data_context = DataContext()
        self.language_model = ModelContext()

    def chatbot_response(self, msg):
        prediction = self.language_model.predict(msg)
        data = self.data_context.get_data()
        return self._get_response(prediction, data)

    @staticmethod
    def _get_response(intents_list, intents_json):
        if len(intents_list) == 0:
            return "Sorry! I don't understand."
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for intent in list_of_intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "Sorry! I don't understand."

    def create_dataset(self, file_path):
        self.data_context.preprocessing_file(file_path)

    def train_model(self):
        if self.data_context.is_json_data():
            data = self.data_context.get_data()
            self.language_model.train(data)
            print("Wytrenowano model")
        else:
            print("Blad przy treningu modelu")

    def is_model_trained(self):
        return self.language_model.is_trained()
