# import glob
# from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from kneed import KneeLocator
import pickle
import warnings
warnings.filterwarnings('ignore')

# All preprocessing steps
import os
# load the data
import pandas as pd

class predPreprocessor:
    def __init__(self, data):
        self.data=data
    def dataPreprocessring(self):

        X = self.data
        model_directory=os.getcwd() + '\\models'
        with open(model_directory + '\\' + 'kmeans' + '.sav', 'rb') as f:
            kmeans = pickle.load(f)
        kmeans.fit(X)
        pred = kmeans.predict(X)

        scaler = StandardScaler()
        scaledX = scaler.fit_transform(X)
        scaledX_df = pd.DataFrame(scaledX, columns=X.columns)
        scaledX_df['cluster']=pred
        finalDf = scaledX_df
        return finalDf