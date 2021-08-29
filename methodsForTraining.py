import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
from appLogging import App_Logger
import warnings
warnings.filterwarnings('ignore')

# All training steps

#load finalDf from methodsForPreprocessing

gradBoostacc = []
logRegacc = []
KNNacc = []
randacc = []
xgbacc = []
# finalDf=pd.read_csv(os.getcwd()+"/concatedTrainingFile/finalTrainingData.csv")
# print("finalTrainingData.csv is loaded for training.")
class modelTraining():
    def __init__(self, finalDf):
        self.finalDf=finalDf
        self.log_writer=App_Logger()
        self.file_object = open(os.getcwd()+"\\Logs\\modelTrainingLogs.txt", 'a+')

    def methodForModelTraining(self):

        self.log_writer.log(self.file_object, 'Entered into the methodForModelTraining method of modelTraining Class.')
        numOfCluster = len(self.finalDf['cluster'].unique())
        self.log_writer.log(self.file_object, 'Loading preprocessed data clusterwise for training.')

        for i in range(numOfCluster):
            self.log_writer.log(self.file_object, 'Loaded data for cluster {}'.format(i))
            clusterWiseData = self.finalDf[self.finalDf['cluster'] == i]
            features = clusterWiseData.drop(['class', 'cluster'], axis=1)
            labels = clusterWiseData['class']

            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=1 / 3, random_state=9)

            # ========================XGBoostCLF=========================================
            param_grid_xgboost = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 7],
                'n_estimators': [10, 50, 100]
            }

            self.log_writer.log(self.file_object, 'Applying XgBoost Classifier on cluster {}'.format(i))
            xgboostclf = XGBClassifier()
            xgbGrid = GridSearchCV(xgboostclf, param_grid=param_grid_xgboost, verbose=3, cv=2)
            xgbGrid.fit(x_train, y_train)
            lr = xgbGrid.best_params_['learning_rate']
            md = xgbGrid.best_params_['max_depth']
            ne = xgbGrid.best_params_['n_estimators']

            xgb = XGBClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
            xgb.fit(x_train, y_train)
            xgbPred = xgb.predict(x_test)

            if accuracy_score(y_test, xgbPred):
                xgbacc.append(accuracy_score(y_test, xgbPred))
            else:
                xgbacc.append(roc_auc_score(y_test, xgbPred))
            # ===============================================================================

            # =========================RandomForestCLF=======================================
            param_grid = {"n_estimators": [10, 50, 100], "criterion": ['gini', 'entropy'],
                          "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']
                          }
            self.log_writer.log(self.file_object, 'Applying Random Classifier on cluster {}'.format(i))
            rndClf = RandomForestClassifier()
            gridForest = GridSearchCV(rndClf, param_grid=param_grid, cv=2, verbose=3)
            gridForest.fit(x_train, y_train)
            crt = gridForest.best_params_['criterion']
            mxd = gridForest.best_params_['max_depth']
            mf = gridForest.best_params_['max_features']
            nest = gridForest.best_params_['n_estimators']
            rndClf = RandomForestClassifier(criterion=crt, max_depth=mxd, max_features=mf, n_estimators=nest)
            rndClf.fit(x_train, y_train)
            randPred = rndClf.predict(x_test)
            if accuracy_score(y_test, randPred):
                randacc.append(accuracy_score(y_test, randPred))
            else:
                randacc.append(roc_auc_score(y_test, randPred))

            # =================================================================================

            # =============================LogisticRegression==================================
            self.log_writer.log(self.file_object, 'Applying Logistic Regression on cluster {}'.format(i))
            logReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            params = {
                'C': [0, 0.001, 0.01, 0.1]
            }
            grid = GridSearchCV(logReg, param_grid=params, cv=5, verbose=3)
            grid.fit(x_train, y_train)
            C_ = grid.best_params_['C']
            logReg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=C_)
            logReg.fit(x_train, y_train)

            logRegPred = logReg.predict(x_test)

            if accuracy_score(y_test, logRegPred):
                logRegacc.append(accuracy_score(y_test, logRegPred))
            else:
                logRegacc.append(roc_auc_score(y_test, logRegPred))

            # =================================================================================

            # ========================GradBoostCLF=========================================
            param_grid_gradBoost = {
                'learning_rate': [0.5, 0.1, 0.01],
                'max_depth': [3, 5, 7],
                'n_estimators': [50, 100],
                #                 "max_features":["log2","sqrt"],
                #                 "criterion": ["friedman_mse",  "mae"],
            }
            self.log_writer.log(self.file_object, 'Applying Gradient Boosting Classifier on cluster {}'.format(i))
            gradBoostclf = GradientBoostingClassifier()
            gradGrid = GridSearchCV(gradBoostclf, param_grid=param_grid_gradBoost, verbose=3, cv=2)
            gradGrid.fit(x_train, y_train)
            lr = gradGrid.best_params_['learning_rate']
            md = gradGrid.best_params_['max_depth']
            ne = gradGrid.best_params_['n_estimators']
            #     mf=gradGrid.best_params_['max_features']
            #     crt=gradGrid.best_params_['criterion']

            gradBoostclf = GradientBoostingClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
            gradBoostclf.fit(x_train, y_train)
            gradBoostPred = gradBoostclf.predict(x_test)

            if accuracy_score(y_test, gradBoostPred):
                gradBoostacc.append(accuracy_score(y_test, gradBoostPred))
            else:
                gradBoostacc.append(roc_auc_score(y_test, gradBoostPred))
                # ===============================================================================


            # ==================================KNNCLF=======================================
            self.log_writer.log(self.file_object, 'Applying KNeighbors Classifier on cluster {}'.format(i))
            knnClf = KNeighborsClassifier()
            params_knn = {
                'n_neighbors': [3, 4, 5, 6, 7, 8],
                'leaf_size': [1, 3, 5],
                'algorithm': ['auto', 'kd_tree']
            }
            gridKNN = GridSearchCV(knnClf, param_grid=params_knn, cv=2, verbose=3, n_jobs=-1)
            gridKNN.fit(x_train, y_train)
            nn = gridKNN.best_params_['n_neighbors']
            ls = gridKNN.best_params_['leaf_size']
            algo = gridKNN.best_params_['algorithm']

            knnClf = KNeighborsClassifier(n_neighbors=nn, leaf_size=ls, algorithm=algo)
            knnClf.fit(x_train, y_train)
            knnClfPred = knnClf.predict(x_test)

            if accuracy_score(y_test, knnClfPred):
                KNNacc.append(accuracy_score(y_test, knnClfPred))
            else:
                KNNacc.append(roc_auc_score(y_test, knnClfPred))

            # =================================================================================

        temp = np.stack((logRegacc, gradBoostacc, randacc, xgbacc, KNNacc), axis=1)
        accDf = pd.DataFrame(temp, columns=['logReg', 'gradBoostclf', 'rndClf', 'xgboostclf', 'knnClf'])
        accDf.idxmax(axis=1)
        accDf.to_csv(os.getcwd() + '\\modelsAccuracyFile' + '.csv', index=False, header=True)
        # print(accDf)
        listOfModelNames=list(accDf.idxmax(axis=1))

        self.log_writer.log(self.file_object, 'Saving the models.')
        model_directory = os.getcwd() + '\\models'
        import pickle
        with open(model_directory + '\\' + 'kmeans' + '.sav', 'rb') as f:
            kmeans=pickle.load(f)

        for f in os.listdir(model_directory):
            os.remove(os.path.join(model_directory, f))

        with open(model_directory + '\\' + 'kmeans' + '.sav', 'wb') as f:
            pickle.dump(kmeans, f)


        listOfModels=[logReg, gradBoostclf,rndClf,xgboostclf,knnClf]
        i=0
        for model in listOfModels:
            if i<len(listOfModelNames):
                with open(model_directory +'\\' + listOfModelNames[i]+str(i) +'.sav','wb') as f:
                    pickle.dump(model, f) #save the model to file
            i=i+1
        self.log_writer.log(self.file_object, 'Training Completed Successfully.')
        self.file_object.close()