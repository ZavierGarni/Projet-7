from flask import Flask, jsonify, request
import pandas as pd
import pickle

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


path = 'model.pkl'

with open(path, 'rb') as f2:
    print("utilisation modele lgbm")
    model = pickle.load(f2)

df = pd.read_csv('application_train_clean.csv')
df.drop(labels='TARGET', axis=1, inplace=True)
num_client = df.SK_ID_CURR.unique()

def predict_id(ID, dataframe):
    '''Fonction de prédiction utilisée par l\'API flask :
    a partir de l'identifiant et du jeu de données
    renvoie la prédiction à partir du modèle'''

    ID = int(ID)
    X = dataframe[dataframe['SK_ID_CURR'] == ID]
    prediction = model.predict(X.drop(labels="SK_ID_CURR", axis=1))
    proba = model.predict_proba(X.drop(labels="SK_ID_CURR", axis=1))
    #print(proba)
    #print(proba[0])
    #print(proba[0][0])
    if proba[0][0] > 0.48:
        return 0, proba
    else:
        return 1, proba

    return prediction, proba


@app.route('/')
def hello():
    return 'Hello'



@app.route('/credit/<int:id_client>', methods=['GET'])
def credit(id_client):

    #récupération id client depuis argument url
    id_client = request.args.get('id_client', default=1, type=int)
    
    #DEBUG
    #print('id_client : ', id_client)
    #print('shape df ', df.shape)
    
    #calcul prédiction défaut et probabilité de défaut
    prediction, proba = predict_id(id_client, df)

    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)


if __name__ == '__main__':
	app.run()
					 
