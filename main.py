import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

bank = pandas.read_csv("bank.csv", sep=";")
bank = bank.drop(columns=["age", "education", "default", "contact", "day", "month", "duration", "pdays", "previous"])

bank = pandas.get_dummies(bank, columns=["job", "marital", "poutcome", "housing", "loan"], dtype=int)

bank["y"] = bank["y"].map({"no": 0, "yes": 1}).astype(int)
            
t_x = bank.drop(columns=["y"])
t_y = bank.loc[:, ["y"]]

x_train, x_test, y_train, y_test = train_test_split(t_x, t_y, test_size=0.2, random_state=50)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

modele = LogisticRegression(max_iter=1000)
modele.fit(x_train, y_train)

y_pred = modele.predict(x_test)

score = metrics.accuracy_score(y_test, y_pred)
print("Précision du modèle :", score*100, "%")