from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    N=float(request.form.get('N'))
    P = float(request.form.get('P'))
    K = float(request.form.get('K'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph= float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))
    data = pd.read_csv('Crop_recommendation.csv')
    data.dropna()
    ss = StandardScaler()
    y = data['label']
    x = data.drop('label', axis=1)
    ss.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
    model = GaussianNB()
    model.fit(x_train, y_train)
    a=[N, P, K, temperature, humidity, ph, rainfall]
    result=model.predict([a])
    data=data[data.label !=result[0]]
    y = data['label']
    x = data.drop('label', axis=1)
    ss.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
    model.fit(x_train, y_train)
    result1 = model.predict([a])
    data = data[data.label != result1[0]]
    y = data['label']
    x = data.drop('label', axis=1)
    ss.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)
    model.fit(x_train, y_train)
    result2 = model.predict([a])


    return render_template('result.html',**locals())
if __name__== '__main__':
    app.run(debug=True,port=5500)
