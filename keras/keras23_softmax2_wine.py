# acc : 0.95
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


#1. 데이터

datasets = load_wine()
print(datasets)
x = datasets.data
y = datasets.target

print('x.shape :', x.shape)
print(np.unique(y, return_counts=True))

y = to_categorical(y)
# print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=333, stratify=y)

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor="val_loss",
    patience = 50,
    mode='auto',
    restore_best_weights=True,
)

hist = model.fit(x_train, y_train, epochs=2000, validation_split=0.2,
                 batch_size=8,
                 callbacks=[es],
                 )

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss :', result[0])
print('acc :', result[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc :', acc)

# loss : 0.17151986062526703
# acc : 0.9444444179534912
# acc : 0.9444444444444444

# loss : 0.14503976702690125
# acc : 0.9444444179534912
# acc : 0.9444444444444444