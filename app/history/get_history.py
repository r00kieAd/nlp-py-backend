import json, os, logging
import traceback, pickle

class Get_History():

    def __init__(self):
        self.history_path = os.path.join(os.path.dirname(__file__), 'training_history.pkl')
        self.last_epoch = 0
        self.total_epochs = 0
        self.accuracy = 0
        self.loss = 0
        self.date = ''
    
    def loadData(self):
        try:
            with open(self.history_path, "rb") as file:
                history_data = pickle.load(file)
                self.last_epoch = history_data["last_epoch_count"]
                self.total_epochs = history_data["total_epochs"]
                self.date = history_data["date"]
                self.accuracy = f'{history_data["history"]["accuracy"][-1] * 100:.4f}%'
                self.loss = f'{history_data["history"]["loss"][-1] * 100:.4f}%'
            return True
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            return {"error": f"{str(e)}, 'Line:', {line_no}", "status": "failed"}
    
    def getData(self):
        try:
            data_loaded = self.loadData()
            if isinstance(data_loaded, bool):
                return {
                    "status": "success",
                    "last_training_date": self.date,
                    "last_epoch_count": self.last_epoch,
                    "total_epochs": self.total_epochs,
                    "model_accuracy": self.accuracy,
                    "model_loss": self.loss
                }
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            return {"error": f"{str(e)}, 'Line:', {line_no}", "status": "failed"}