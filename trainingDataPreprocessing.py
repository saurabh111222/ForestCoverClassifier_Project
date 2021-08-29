# import glob
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.cluster import KMeans
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')
import os
from appLogging import App_Logger

# All preprocessing steps
# load the data
import pandas as pd

class preprocessor:
    def __init__(self):
        self.log_writer=App_Logger()
        self.file_object=open(os.getcwd()+"\\Logs\\dataPreprocessingLogs.txt", 'a+')
    def dataPreprocessring(self, data):

        self.log_writer.log(self.file_object, 'Entered into the dataPreprocessing method of preprocessor class.')
        self.log_writer.log(self.file_object, 'Creating upper and lower boundary for the removal of outliers.')
        lowerBoundary = data['Horizontal_Distance_To_Fire_Points'].mean() - 3 * data['Horizontal_Distance_To_Fire_Points'].std()
        upperBoundary = data['Horizontal_Distance_To_Fire_Points'].mean() + 3 * data['Horizontal_Distance_To_Fire_Points'].std()
        data.loc[data['Horizontal_Distance_To_Fire_Points'] >= upperBoundary, 'Horizontal_Distance_To_Fire_Points'] = upperBoundary
        data.loc[data['Horizontal_Distance_To_Fire_Points'] <= lowerBoundary, 'Horizontal_Distance_To_Fire_Points'] = lowerBoundary
        self.log_writer.log(self.file_object, 'Outlier removed.')
        X = data.drop('class', axis=1)
        Y = data['class']

        oversampler = SMOTE()
        X, Y = oversampler.fit_resample(X, Y)
        oversampled_df = pd.concat([X, Y], axis=1)
        self.log_writer.log(self.file_object, 'Applied SMOTE to balance the dataset.')

        self.log_writer.log(self.file_object, 'Initializing kmeans.')
        WCSS = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', n_jobs=-1)
            kmeans.fit(X)
            WCSS.append(kmeans.inertia_)

        kn = KneeLocator(range(1, 11), WCSS, curve='convex', direction='decreasing')
        value_of_K = kn.knee
        self.log_writer.log(self.file_object, "KMeans applied and the value of K is {}".format(value_of_K))

        kmeans = KMeans(n_clusters=value_of_K, init='k-means++', n_jobs=-1)
        kmeans.fit(X)
        pred = kmeans.predict(X)
        oversampled_df['cluster'] = pred

        self.log_writer.log(self.file_object, 'Saving Kmeans model')
        pathForKmeans=os.getcwd()+'\\models'
        with open(pathForKmeans + '\\' + 'kmeans' + '.sav', 'wb') as f:
            pickle.dump(kmeans, f)
        self.log_writer.log(self.file_object, 'Kmeans model saved')

        oversampled_df['class'] = oversampled_df['class'].map(
            {'Spruce_Fir': 1, 'Lodgepole_Pine': 2, 'Ponderosa_Pine': 3, 'Cottonwood_Willow': 4, 'Aspen': 5,
             'Douglas_fir': 6, 'Krummholz': 7})
        X = oversampled_df.drop(['class', 'cluster'], axis=1)
        Y = oversampled_df[['class', 'cluster']]

        self.log_writer.log(self.file_object, 'Initializing standardScaler.')
        scaler = StandardScaler()
        scaledX = scaler.fit_transform(X)
        scaledX_df = pd.DataFrame(scaledX, columns=X.columns)
        finalDf = pd.concat([scaledX_df, Y], axis=1)
        self.log_writer.log(self.file_object, 'Data Preprocessing Sucessfully Completed.')
        self.file_object.close()
        return finalDf
