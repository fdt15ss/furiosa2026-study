from sklearn.datasets import load_digits
# acc : 1.0
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

ohe = OneHotEncoder(sparse_output=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)

print(f'x.shape : {x.shape}, y.shape : {y.shape}')
print(f'x.dtype : {x.dtype}, y.dtype : {y.dtype}')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   mode='auto',
                   restore_best_weights=True
                   )

hist = model.fit(x_train, y_train,
                 epochs=1000, batch_size=128,
                 validation_split=0.2,
                 callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)
acc_categorical = np.mean(categorical_accuracy(y_test, y_predict))
print('acc_categorical :', acc_categorical)

# loss :  0.07446880638599396
# acc :  0.9916666746139526
# acc_categorical : 0.9916667
# loss(MinMaxScaler 적용) :  0.20266978442668915
# acc(MinMaxScaler 적용) :  0.949999988079071
# acc_categorical(MinMaxScaler 적용) : 0.95