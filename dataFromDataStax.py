import os

import cassandra
print(cassandra.__version__)
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
from schemaFinder import selectSchema
from cassandra.query import dict_factory
from appLogging import App_Logger
# from cassandra.query import tuple_factory
class dataStaxAstra():
    def __init__(self):
        self.log_writer=App_Logger()
        self.file_object = open(os.getcwd()+"\\Logs\\dataStaxLog.txt", 'a+')

    def loadDataFromDataStax(self):
        cloud_config= {
                'secure_connect_bundle': os.getcwd()+'\\secure-connect-forestCoverClassisfication.zip'
        }
        auth_provider = PlainTextAuthProvider('pwntZGjzQAwcbOhkdyzujzTu', ',8.3t6H_Pf9YF2-D8RZi6LYbK6kZGSlImkuM-,RJsC8LdpQL30U3xMtwGH-wZhj6gURmPAZ3PTuhKRk4DGdHE5SoUBdR0z3T-x9Cq97OpZL8KBW6SbuKfNQoWvA7ZjH9')
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session = cluster.connect()

        row = session.execute("select release_version from system.local").one()
        if row:
            print(row[0])
            self.log_writer.log(self.file_object, 'Connection established.')
        else:
            self.log_writer.log(self.file_object, 'An error occurred while connecting the DataStax Astra Cluster')
            print("An error occurred.")

        session = cluster.connect('ForestCoverClassifier')
        self.log_writer.log(self.file_object, 'Connected to dataStax Astra Cluster.')
        session.row_factory = dict_factory

        #list of data files for training
        listOfTablesInKeyspace=['datafortraining_1'] #, 'datafortraining_1'
        for tableName in listOfTablesInKeyspace:
            rows = session.execute("SELECT * FROM " + tableName)
            dictList=list(rows)
            df=pd.DataFrame(dictList)
            # print()
            obj = selectSchema()
            schemaForTraining =obj.trainingSchema()
            df=df[schemaForTraining]

            pathToDataFromDtax=os.getcwd()+'\\TrainingDataFilesFromDataStax\\{}{}'.format(tableName, '.csv')
            df.to_csv(pathToDataFromDtax, encoding='utf-8', index=False)
            self.log_writer.log(self.file_object,"Training file: {} is loaded into dir: TrainingDataFilesFromDataStax ".format(tableName))

        #list of data files for training
        predictionFile='dataforprediction_1'
        rows = session.execute("SELECT * FROM " + predictionFile)
        dictList=list(rows)
        df=pd.DataFrame(dictList)
        predObj=selectSchema()
        schemaForPrediction = predObj.predSchema()
        df=df[schemaForPrediction]
        pathToPredictionDataFilesFromDataStax=os.getcwd()+'\\PredictionDataFilesFromDataStax\\{}{}'.format(predictionFile, '.csv')
        df.to_csv(pathToPredictionDataFilesFromDataStax, encoding='utf-8', index=False)
        self.log_writer.log(self.file_object,"Prediction file: {} is loaded into dir: TrainingDataFilesFromDataStax ".format(predictionFile))

        cluster.shutdown()
        self.log_writer.log(self.file_object, 'DataStax Astra Cluster closed.')
        self.file_object.close()
