import numpy as np
import pandas as pd
import pickle
import flask

with open(f'model/linear_dg.pkl', 'rb') as file:
    model = pickle.load(file)

app = flask.Flask(__name__, template_folder='template')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        test = model.theta
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        LotFrontage = flask.request.form['LotFrontage']
        MasVnrArea = flask.request.form['MasVnrArea']
        TotalBsmtSF = flask.request.form['TotalBsmtSF']
        firstFlrSF = flask.request.form['firstFlrSF']
        GrLivArea = flask.request.form['GrLivArea']
        FullBath = flask.request.form['FullBath']
        TotRmsAbvGrd = flask.request.form['TotRmsAbvGrd']
        Fireplaces = flask.request.form['Fireplaces']
        GarageCars = flask.request.form['GarageCars']
        GarageArea = flask.request.form['GarageArea']
        MasVnrType = flask.request.form['MasVnrType']
        OverallQual = flask.request.form['OverallQual']
        ExterQual = flask.request.form['ExterQual']
        BsmtQual = flask.request.form['BsmtQual']
        HeatingQC = flask.request.form['HeatingQC']
        KitchenQual = flask.request.form['KitchenQual']
        FireplaceQu = flask.request.form['FireplaceQu']
        YearBuilt = flask.request.form['YearBuilt']
        YearRemodAdd = flask.request.form['YearRemodAdd']
        GarageYrBlt = flask.request.form['GarageYrBlt']
        Age = flask.request.form['Age']

        inputs = pd.DataFrame(
            columns=['LotFrontage', 'MasVnrArea', 'TotalBsmtSF', 'firstFlrSF', 'GrLivArea', 'FullBath',
                     'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'MasVnrType', 'OverallQual',
                     'ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'YearBuilt',
                     'YearRemodAdd', 'GarageYrBlt', 'Age'],
            data=[LotFrontage, MasVnrArea, TotalBsmtSF, firstFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, Fireplaces,
                  GarageCars, GarageArea, MasVnrType, OverallQual, ExterQual, BsmtQual, HeatingQC, KitchenQual,
                  FireplaceQu, YearBuilt, YearRemodAdd, GarageYrBlt, Age])
        prediction = model.predict(inputs)

        return flask.render_template('index.html', result=prediction,)


if __name__ == '__main__':
    app.run()
