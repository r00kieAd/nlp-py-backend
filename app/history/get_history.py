import json, os, logging
import traceback, pickle

class Get_History():

    def __init__(self):
        self.history_path = os.path.join(os.path.dirname(__file__), 'training_history.pkl')
        self.last_epoch1 = 0
        self.total_epochs1 = 0
        self.last_epoch2 = 10
        self.total_epochs2 = 30
        self.accuracy1 = 0
        self.loss1 = 0
        self.accuracy2 = 0
        self.loss2 = 0
        self.date = ''
    
    def loadData(self):
        try:
            with open(self.history_path, "rb") as file:
                history_data = pickle.load(file)
                self.last_epoch1 = history_data["last_epoch_count"]
                self.total_epochs1 = history_data["total_epochs"]
                self.last_epoch2 = history_data["last_transformer_epoch"]
                self.total_epochs2 = history_data["total_transformer_epochs"]
                self.date = history_data["date"]
                self.accuracy1 = f'{history_data["history"]["accuracy"][-1] * 100:.4f}%'
                self.loss1 = f'{history_data["history"]["loss"][-1] * 100:.4f}%'
                self.accuracy2 = f'{history_data["transformer_history"]["accuracy"][-1] * 100:.4f}%'
                self.loss2 = f'{history_data["transformer_history"]["loss"][-1] * 100:.4f}%'
            return True
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            return {"error": f"{str(e)}, 'Line:', {line_no}", "status": "failed"}
    
    def getData(self):
        try:
            data_loaded = self.loadData()
            print(data_loaded)
            if isinstance(data_loaded, bool):
                print({"status": "success",
                    "last_training_date": self.date,
                    "last_epoch_count": self.last_epoch1,
                    "total_epochs": self.total_epochs1,
                    "model_accuracy": self.accuracy1,
                    "model_loss": self.loss1,
                    "last_transformer_epochs": self.last_epoch2,
                    "total_transformer_epochs": self.total_epochs2,
                    "transformer_accuracy": self.accuracy2,
                    "transformer_loss": self.loss2})
                return {
                    "status": "success",
                    "last_training_date": self.date,
                    "last_epoch_count": self.last_epoch1,
                    "total_epochs": self.total_epochs1,
                    "model_accuracy": self.accuracy1,
                    "model_loss": self.loss1,
                    "last_transformer_epochs": self.last_epoch2,
                    "total_transformer_epochs": self.total_epochs2,
                    "transformer_accuracy": self.accuracy2,
                    "transformer_loss": self.loss2
                }
        except Exception as e:
            line_no = traceback.extract_tb(e.__traceback__)[-1].lineno
            return {"error": f"{str(e)}, 'Line:', {line_no}", "status": "failed"}