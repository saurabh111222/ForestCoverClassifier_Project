# imports
import glob
import pandas as pd
from trainingDataPreprocessing import preprocessor
from predictionDataPreprocessing import predPreprocessor
from dataFromDataStax import dataStaxAstra
from methodsForTraining import modelTraining
from methodsForPrediction import prediction
import os
from wsgiref import simple_server
import flask_monitoringdashboard as dashboard
from flask import Flask, request, render_template, Response, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


extensions = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions
# route for home
@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

# route for training
@app.route('/train', methods=['GET'])
@cross_origin()
############################# Code for loading the raw data and call the preprocessing method on it #################################
# loading data for training
def modelPreproAndTrain():
    #code for load the data from DataStax Astrs
    dataStaxObj=dataStaxAstra()
    dataStaxObj.loadDataFromDataStax()
    print("All the data files including training and prediction loaded successfully from DataStax Astra in their respecctive folder directories.")
    #==============================================================================================================================================

    path = os.getcwd()+'\\TrainingDataFilesFromDataStax'
    fileNames = [f for f in glob.glob(path + "/*.csv")]
    for i, csvFilePath in enumerate(fileNames):
        csvFile=pd.DataFrame(pd.read_csv(csvFilePath))
        csvFile.to_csv(os.getcwd()+'\\dataFilesForTraining'+'\\datafortraining_'+str(i+1)+'.csv', index=False)
        print('Imported CSVs dataFilesForTraining')
    print("All training files imported in dataFilesForTraining folder.")

    filePath=(os.getcwd()+'\\dataFilesForTraining\\datafortraining_1.csv')
    data=pd.read_csv(filePath)
    ValObj=preprocessor()
    preprocessedData=ValObj.dataPreprocessring(data)
    print('Preprocessing done')

    # call for training and now use preprocessedData for training and do not save preprocessed data
    trainObj=modelTraining(preprocessedData)
    trainObj.methodForModelTraining()
    print("Training successfully completed.")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# route for prediction

@app.route('/predict', methods=['POST'])
@cross_origin()
############################# Code for loading the raw prediction data and call the preprocessing method it #################################
# loading data for prediction
def modelPreproPrediction():
    # path=request.form['filepath']
    path = os.getcwd()+'\\PredictionDataFilesFromDataStax' # use your path
    # predfileName=glob.glob(path + "/*.csv")
    rawPredFile=pd.read_csv(path+'\\dataforprediction_1.csv')
    rawPredFile.to_csv(os.getcwd()+'\\filesForPrediction\\finalRawPredictionData.csv', index=False)

    # call the preprocessing of prediction data
    data=pd.read_csv(os.getcwd()+"\\filesForPrediction\\finalRawPredictionData.csv")
    predValObj=predPreprocessor(data)
    dataForPred=predValObj.dataPreprocessring()

    # call the prediction
    predObj=prediction()
    predObj.predictionMethod(dataForPred)
    df1 = pd.read_csv(r'C:\Users\LENOVO\Desktop\datastax\predictionOutputFile\Predictions.csv')
    df1 = df1.drop(df1.columns[0], axis=1)
    df2 = pd.read_csv(r'C:\Users\LENOVO\Desktop\datastax\PredictionDataFilesFromDataStax\dataforprediction_1.csv')
    df2 = df2[['elevation','aspect', 'slope', 'horizontal_distance_to_hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']]
    concatDf = pd.concat([df2, df1], axis=1)
    tuplesOfdf = tuple(concatDf.to_records(index=False))
    header = tuple(concatDf.columns)
    return render_template('/table.html', headings=header, data=tuplesOfdf)
    # filepath=os.getcwd()+'\\predictionOutputFile'
    # return Response("Prediction file created at %s " %filepath)




@app.route('/customPredict', methods=['GET'])
@cross_origin()
############################# Code for loading the custom uploaded prediction data and call the preprocessing method it #################################
# loading data for prediction
def customPredict():
    path = os.getcwd()+'\\uploads_folder' # use your path
    customFilename = session.get('customFilename', None)
    data = pd.read_csv(path +'\\' +customFilename)
    predValObj=predPreprocessor(data)
    dataForPred=predValObj.dataPreprocessring()

    # call the prediction
    predObj=prediction()
    predObj.predictionMethod(dataForPred)
    df1 = pd.read_csv(r'C:\Users\LENOVO\Desktop\datastax\predictionOutputFile\Predictions.csv')
    df1 = df1.drop(df1.columns[0], axis=1)
    df2 = pd.read_csv(r'C:\Users\LENOVO\Desktop\datastax\uploads_folder\dataforprediction_1.csv')
    concatDf = pd.concat([df2, df1], axis=1)
    tuplesOfdf = tuple(concatDf.to_records(index=False))
    header = tuple(concatDf.columns)
    # header=('Index','Class')
    return render_template('/table.html', headings=header, data=tuplesOfdf)

@app.route('/upload', methods=['POST', 'GET'])
@cross_origin()
def upload_file():
    upload_dest = os.path.join(os.getcwd(), 'uploads_folder')

    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No files found, try again.')
            return redirect(request.url)
    files = request.files.getlist('files[]')
    # filename=''
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join( upload_dest, filename))
            session['customFilename']=filename
    flash('File(s) uploaded')
    return redirect(url_for('customPredict'))


if __name__=='__main__':
    app.run(debug=True)
    host='0.0.0.0'
    port=5000
    httpd=simple_server.make_server(host, port, app)
    httpd.serve_forever()