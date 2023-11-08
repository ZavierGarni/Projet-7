#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import pickle
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap


#Load Dataframe
path_df = '/Users/zavier/Desktop/Data_Scientist_OC/Projet_7_Implémentez_un_modèle_de_scoring_Garni_Xavier/application_train_clean.csv'
path_rawdf = '/Users/zavier/Desktop/Data_Scientist_OC/Projet_7_Implémentez_un_modèle_de_scoring_Garni_Xavier/application_train.csv'


path = '/Users/zavier/Desktop/Data_Scientist_OC/Projet_7_Implémentez_un_modèle_de_scoring_Garni_Xavier/mlruns/models/LGBM_hyperopt/model.pkl'

with open(path, 'rb') as f2:
    print("utilisation modele lgbm")
    model = pickle.load(f2)



@st.cache_data #mise en cache de la fonction pour exécution unique
def chargement_data(path):
    dataframe = pd.read_csv(path)
    return dataframe


@st.cache_data #mise en cache de la fonction pour exécution unique
def chargement_ligne_data(id, df):
    return df[df['SK_ID_CURR']==int(id)].drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)


@st.cache_data
def calcul_valeurs_shap(df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.drop(labels=["SK_ID_CURR","TARGET"], axis=1))
    return shap_values






dataframe = chargement_data(path_df)
raw_dataframe = chargement_data(path_rawdf)
liste_id = dataframe['SK_ID_CURR'].tolist()


y_pred_lgbm = model.predict(dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1))    # Prédiction de la classe 0 ou 1
y_pred_lgbm_proba = model.predict_proba(dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1)) # Prédiction du % de risque

# Récupération du score du client
y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                dataframe['SK_ID_CURR']], axis=1)

















# Affichage du formulaire
st.title('Prêt à dépenser :bar_chart:')
st.subheader("De quel client voulez-vous connaître le résultat ?")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client :')
#chaine = "l'id Saisi est " + str(id_input)
#st.write(chaine)


sample_en_regle = str(list(dataframe[dataframe['TARGET'] == 0].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(dataframe[dataframe['TARGET'] == 1].sample(5)[['SK_ID_CURR', 'TARGET']]['SK_ID_CURR'].values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples de clients en défaut : ' + sample_en_defaut

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_regle)
    st.write(chaine_en_defaut)


elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API

    # Appel de l'API : 

    API_url = "http://127.0.0.1:5000/credit/" + id_input
    info_client = requests.get(API_url, params={"id_client":id_input}).json()

    classe_predite = info_client['prediction']
    if classe_predite == 1:
        etat = 'client à risque'
    else:
        etat = 'client peu risqué'
    proba = 1-info_client['proba'] 

    # Affichage de la prédiction
    prediction = info_client['proba']
    classe_reelle = int(dataframe[dataframe['SK_ID_CURR']==int(id_input)]['TARGET'].values[0])
    classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
    chaine1 = f'Prédiction : {(1-prediction)*100:.2f}% de risque de défaut'
    chaine2 = '(Classe réelle : '+ str(classe_reelle) + ')'

    st.markdown(chaine1)
    st.markdown(chaine2)
    
    row_df_sk = (dataframe['SK_ID_CURR'] == int(id_input))
    row_appli_sk = (raw_dataframe['SK_ID_CURR'] == int(id_input))
    
    # Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1))

    # récupération de l'index correspondant à l'identifiant du client
    idx = int(row_df_sk.index[0])
    
    
    
    
# Impression du graphique jauge
    st.markdown("""---""")
    st.text("")
    
    fig = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = float(prediction),
                    mode = "gauge+number",
                    title = {'text': "Score du client", 'font': {'size': 24}},
                    gauge = {'axis': {'visible': False}}))

    fig.update_layout(paper_bgcolor='white',
                    height=400, width=500,
                    font={'color': '#292929', 'family': 'Roboto Condensed'},
                    margin=dict(l=30, r=30, b=5, t=5))
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"**Il y a donc un risque de {(1-prediction)*100:.2f}% que le client ait des difficultés de paiement.**")
    
    
    
    
    
    
    ###########################################################################
    # Affichage des infos personnelles du client ##############################
    
    st.markdown("""---""")
    
    sex = raw_dataframe.loc[row_appli_sk, ['CODE_GENDER']].values[0][0]
    age = int(np.trunc(- int(raw_dataframe.loc[row_appli_sk, ['DAYS_BIRTH']].values)/365))
    family = raw_dataframe.loc[row_appli_sk, ['NAME_FAMILY_STATUS']].values[0][0]
    education = raw_dataframe.loc[row_appli_sk, ['NAME_EDUCATION_TYPE']].values[0][0]
    occupation = raw_dataframe.loc[row_appli_sk, ['OCCUPATION_TYPE']].values[0][0]
    revenus = raw_dataframe.loc[row_appli_sk, ['AMT_INCOME_TOTAL']].values[0][0]

    
    check = st.checkbox('Afficher les informations personnelles du client')
    if check :
        st.subheader("Informations personnelles")
        st.write("Genre :",sex)
        st.write("Âge :", age)
        st.write("Statut familial :", family)
        st.write("Niveau académique :", education)
        st.write("Emploi :", occupation)
        st.write("Revenu annuel : ", revenus)




    ###########################################################################
    # Explication de la prédiction du client ##################################
    
    st.markdown("""---""")

    # Graphique force_plot
    check2 = st.checkbox('Afficher l\'explication du calcul du score')
    if check2 :
        st.subheader("Comment le score du client est-il calculé ?")
        st.write("Nous observons sur le graphique suivant, quelles sont les variables qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")
        st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                                    shap_values[1][idx,:], 
                                    dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1).iloc[idx,:], 
                                    link='logit',
                                    figsize=(20, 8),
                                    ordering_keys=True,
                                    text_rotation=0,
                                    contribution_threshold=0.05))
    
    # Graphique decision_plot
        st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
            Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
            avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
            “la trajectoire” prise par la prédiction du client pour chacune des valeurs des variables affichées. ")
        st.write("Seules les 15 variables explicatives les plus importantes sont affichées par ordre décroissant.")
        st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1).iloc[idx,:], 
                            feature_names=dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -16, -1), # affichage des 15 variables les + importantes
                            link='logit'))









    ###########################################################################
    # Affichage de la comparaison avec les autres clients #####################
    
    st.markdown("""---""")
    
    shap_values_df = pd.DataFrame(data=shap_values[1], columns=dataframe.drop(labels=["SK_ID_CURR","TARGET"], axis=1).columns)
    
    df_groupes = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], shap_values_df], axis=1)
    df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
                                              q=5,
                                              precision=1,
                                              labels=['20%_et_moins',
                                                      '21%_30%',
                                                      '31%_40%',
                                                      '41%_60%',
                                                      '61%_et_plus'])

    check3 = st.checkbox('Afficher la comparaison du client par rapport aux groupes de clients')
    if check3 :
        st.subheader('Comparaison du client par rapport aux groupes de clients')
    
        # Moyenne des variables par classe
        df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
        df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()
        df_groupes_mean["index"]=[1,2,3,4,5]
        df_groupes_mean.set_index('index', inplace = True)
        
    
        # dataframe avec shap values du client et des 5 groupes de clients
        comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx], 
                                                df_groupes_mean],
                                                axis = 0)
        comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx, 
                                                              dataframe.iloc[idx, 0],
                                                              comparaison_client_groupe['typologie_clients'])
        # transformation en array
        nmp = comparaison_client_groupe.drop(
                          labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()
    
        fig = plt.figure(figsize=(8, 20))
        st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                                    nmp, 
                                    feature_names=comparaison_client_groupe.drop(
                                                  labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
                                    feature_order='importance',
                                    highlight=0,
                                    legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
                                    plot_color='inferno_r',
                                    legend_location='center right',
                                    feature_display_range=slice(None, -57, -1),
                                    link='logit'))










    ###########################################################################
    # Affichage de l'explication features importance SHAP #####################

    st.markdown("""---""")
    
    check4 = st.checkbox('Afficher l\'interprétation globale des caractéristiques')
    if check4 :
        st.subheader("Explication globale du modèle")

        fig = plt.figure()
        plt.title("Interprétation Globale :\n Diagramme d'Importance des Variables", 
            fontname='Roboto Condensed',
            fontsize=8, 
            fontstyle='italic')
        st_shap(shap.summary_plot(calcul_valeurs_shap(dataframe)[1], 
                            feature_names=dataframe.drop(labels="SK_ID_CURR", axis=1).columns,
                            plot_size=(8, 10),
                            color='#0093FF',
                            plot_type="bar",
                            max_display=56,
                            show = False))
        plt.show()
















    
    
    
    
    
    
    