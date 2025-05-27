import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from itertools import combinations

liste_descripteurs = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
              "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]

non_numerical_descripteurs = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

bank = pandas.read_csv("bank-full.csv", sep=";").sample(n=2000, random_state=42)
bank["y"] = bank["y"].map({"no": 0, "yes": 1}).astype(int)
bank_encoded = pandas.get_dummies(bank, columns=non_numerical_descripteurs, dtype=int)

def test_combinaison(combinaison):
    """
    Teste une combinaison sur un échantillon de 2000 valeurs prises dans bank-full.csv

    :param combinaison: list, la combinaison testée
    :return: float, le score de la combinaison
    """
    colonnes = []
    
    for descripteur in combinaison:
        for col in bank_encoded.columns:
            if descripteur in col and col != "y":
                colonnes.append(col)
                
    t_x = bank_encoded[colonnes]
    t_y = bank_encoded["y"]
    x_train, x_test, y_train, y_test = train_test_split(t_x, t_y, test_size=0.2, random_state=42)
    
    modele = KNeighborsClassifier(n_neighbors=5)
    modele.fit(x_train, y_train)
    
    y_pred = modele.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    return score

def brute_force():
    """
    Teste toutes les combinaisons possible (jusque 5 de longueur max) et affiche la meilleure d'entre elles et son score
    
    :return: affichage
    """
    best_score = 0
    best_combinaison = None
    for i in range(1, 6): #limite la taille des combinaisons à 5 max (sinon trop long)
        for combinaison in combinations(liste_descripteurs, i):
            score = test_combinaison(combinaison)
            if score > best_score:
                best_score = score
                best_combinaison = combinaison
    print("Meilleure combinaison :", best_combinaison)
    print("Score :", round(best_score*100, 2), "%")

brute_force() #temps de calcul ~2min