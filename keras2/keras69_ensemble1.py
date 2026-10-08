import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T     # (100, 2)
                        # 삼성 종가        하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411,511),    # (100, 3)
                            # 원유가            환율
                        range(150, 250)]).transpose()
                            #금시세
y = np.array(range(3001, 3101))
                # 화성의 화씨 온도
    
x1_datasets_train, x1_datasets_test, x2_datasets_train, x2_datasets_test, y_train, y_test = train_test_split(
    x1_datasets,
    x2_datasets,
    y,
    train_size=0.8,
    random_state=42
)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(30, activation= 'relu', name='han1')(input1)
dense2 = Dense(20, activation= 'relu', name='han2')(dense1)
dense3 = Dense(10, activation= 'relu', name='han3')(dense2)
output1 = Dense(5, activation= 'relu', name='han4')(dense3)
# model1 = Model(inputs = input1, outputs = output1)

#2-2. 모델
input21 = Input(shape=(3,))
dense21 = Dense(50, activation= 'relu', name='han21')(input21)
dense22 = Dense(40, activation= 'relu', name='han22')(dense21)
dense23 = Dense(30, activation= 'relu', name='han23')(dense22)
dense24 = Dense(20, activation= 'relu', name='han24')(dense23)
output21 = Dense(3, activation= 'relu', name='han25')(dense24)
# model2 = Model(inputs = input21, outputs=output21)

#2-3. 모델 합치기.
from keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input21], outputs = last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience = 50,
    restore_best_weights=True,
)

model.fit([x1_datasets_train, x2_datasets_train], y_train, epochs=1000, batch_size=32,
          validation_split=0.1,
          callbacks=[es]
          )

#4. 평가, 예측
result = model.evaluate([x1_datasets_test, x2_datasets_test], y_test)
print('loss : ', result)

x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                    range(249, 255)]).T


y_pred = model.predict([x1_pred, x2_pred])
# loss :  1.4176477193832397
print(y_pred)
# [[3096.491 ]
#  [3096.3967]
#  [3099.3032]
#  [3103.8738]
#  [3107.6992]
#  [3112.445 ]]
# 예측값까지 완성
# 맹그러봐

