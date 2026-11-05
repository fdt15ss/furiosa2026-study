import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001, 0.3]
max_depth = [3,4,5,6,7,8]
best_score = 0
best_parameters = ''

for i, lr in enumerate(learning_rate):
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=md,
        )
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = lr, md
        print(i, ",", j, "번째 score :", round(score,4),"도는중...")

print("최고 점수 : {:.2f}".format(best_score))
print("최적 매개변수 :", best_parameters)