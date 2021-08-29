import os
import pickle
global model
class bestModel:
    def __init__(self):
        pass
    def bestModelForData(self, model_directory, clusterNumber):
        listOfFiles = os.listdir(
            model_directory)  # ['gradBoostclf0.sav', 'gradBoostclf2.sav', 'kmeans.sa v', 'xgboostclf1.sav']

        for file in listOfFiles:
            try:
                if (file.index(str(clusterNumber)) != -1):
                    name = file
            except:
                continue

            with open(model_directory + '\\' + name, 'rb') as f:
                model = pickle.load(f)

        return model