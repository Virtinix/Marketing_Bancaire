import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

liste_descripteurs = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
              "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]

non_numerical_descripteurs = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

def test_combinaison(combinaison):
    """
    Teste une combinaison sur un échantillon de 45000 valeurs prises dans bank-full.csv

    :param combinaison: list, la combinaison testée
    :return: affichage
    """
    global t_x
    global modele
    global colonnes_a_encoder
    
    bank = pandas.read_csv("bank-full.csv", sep=";")
    bank["y"] = bank["y"].map({"no": 0, "yes": 1}).astype(int)
    
    colonnes_utiles = combinaison + ["y"]
    bank = bank[colonnes_utiles]

    colonnes_a_encoder = [col for col in combinaison if col in non_numerical_descripteurs]
    
    if colonnes_a_encoder:
        bank = pandas.get_dummies(bank, columns=colonnes_a_encoder, dtype=int)

    t_x = bank.drop(columns=["y"])
    t_y = bank["y"]
    x_train, x_test, y_train, y_test = train_test_split(t_x, t_y, test_size=0.2, random_state=50)
    
    modele = KNeighborsClassifier(n_neighbors=5)
    modele.fit(x_train, y_train)
    
    y_pred = modele.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print("Précision du modèle :", round(score*100, 2), "%")

def predict_client(client_dict):
    """
    Prédit si un client X aurait souscrit un compte de dépôts

    :param client_dict: dict, les données du client X
    :return: str, le résultat de la prédiction
    """
    client = pandas.DataFrame([client_dict])
    
    client = pandas.get_dummies(client, columns=colonnes_a_encoder, dtype=int)
    
    #ajoute les colonnes manquantes pour que client_df ait les mêmes colonnes que les données d'entraînement
    for colonne in t_x.columns:
        if colonne not in client.columns:
            client[colonne] = 0
    client = client[t_x.columns]
    
    prediction = modele.predict(client)
    
    if prediction[0] == 1:
        return "oui"
    else:
        return "non"

#TESTS
client_X = {
    "age": 30,
    "job": "unemployed",
    "marital": "married",
    "education": "primary",
    "default": "no",
    "balance": 1787,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 19,
    "month": "oct",
    "duration": 79,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

client_Y = {
    "age": 21,
    "job": "student",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 2488,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 30,
    "month": "jun",
    "duration": 258,
    "campaign": 6,
    "pdays": 169,
    "previous": 3,
    "poutcome": "success"
}

combinaison = ['job', 'default', 'loan', 'campaign', 'poutcome']

test_combinaison(combinaison)
print("Prédiction pour le client X :", predict_client(client_X))
print("Prédiction pour le client Y :", predict_client(client_Y))