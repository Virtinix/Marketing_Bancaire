import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

bank = pandas.read_csv("bank-full.csv", sep=";")
bank = bank.drop(columns=["age", "education", "default", "contact", "day", "month", "campaign"])

bank = pandas.get_dummies(bank, columns=["job", "marital", "housing", "loan", "poutcome", "previous"], dtype=int)

bank["y"] = bank["y"].map({"no": 0, "yes": 1}).astype(int)
            
t_x = bank.drop(columns=["y"])
t_y = bank["y"]

x_train, x_test, y_train, y_test = train_test_split(t_x, t_y, test_size=0.2, random_state=50)

modele = KNeighborsClassifier(n_neighbors=5)
modele.fit(x_train, y_train)

y_pred = modele.predict(x_test)

score = metrics.accuracy_score(y_test, y_pred)
print("Précision du modèle :", score*100, "%")

def predict_client(client_dict):
    """
    Prédit si un client X aurait souscrit un compte de dépôts

    :param client_dict: dict, les données du client X
    :return: str, le résultat de la prédiction
    """
    client = pandas.DataFrame([client_dict])
    
    client = pandas.get_dummies(client, columns=["job", "marital", "housing", "loan", "poutcome", "previous"], dtype=int)
    
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
    "job": "unemployed",
    "marital": "married",
    "balance": 1787,
    "housing": "no",
    "loan": "no",
    "duration": 79,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

client_Y = {
    "job": "management",
    "marital": "single",
    "balance": 2536,
    "housing": "yes",
    "loan": "no",
    "duration": 958,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

print("Prédiction pour le client X :", predict_client(client_X))
print("Prédiction pour le client Y :", predict_client(client_Y))