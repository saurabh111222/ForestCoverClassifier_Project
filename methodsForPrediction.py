# load the saved model and predict
import os
from findBestModel import bestModel
import pickle
import pandas as pd
from appLogging import App_Logger
from findBestModel import bestModel
class prediction:
    def __init__(self):
        self.log_writer=App_Logger()
        self.file_object = open(os.getcwd()+"\\Logs\\modelPredictionLogs.txt", 'a+')
    def predictionMethod(self, data):
        self.log_writer.log(self.file_object, 'Entered into predictionMethod of prediction class.')
        model_directory = os.getcwd() + '\\models'

        listOfClusters=data['cluster'].unique()
        result=[]
        self.log_writer.log(self.file_object, 'Loading data for prediction clusterwise.')
        for clusterNumber in listOfClusters:
            clusterData=data[data['cluster']==clusterNumber]
            clusterData=clusterData.drop(['cluster'], axis=1)

            self.log_writer.log(self.file_object, 'Finding best model for data belongs to cluster ()'.format(clusterNumber))
            obj=bestModel()
            model=obj.bestModelForData(model_directory, clusterNumber)
            self.log_writer.log(self.file_object, 'Best model found.')
            df=model.predict(clusterData)

            for val in (df):
                if val == 1:
                    result.append("Lodgepole_Pine")
                elif val == 2:
                    result.append("Spruce_Fir")
                elif val == 3:
                    result.append("Douglas_fir")
                elif val == 4:
                    result.append("Krummholz")
                elif val == 5:
                    result.append("Ponderosa_Pine")
                elif val == 6:
                    result.append("Aspen")
                elif val == 7:
                    result.append("Cottonwood_Willow")

        resultDf = pd.DataFrame(result, columns=['Predictions'])

        path=os.getcwd()+'\\predictionOutputFile\\Predictions.csv'
        resultDf.to_csv(path, header=True, mode='w')
        self.log_writer.log(self.file_object, 'Results saved in file Prediction.csv')
        self.log_writer.log(self.file_object, 'Model prediction completed successfully.')
        self.file_object.close()