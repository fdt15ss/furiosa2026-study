# 69-2 카피

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
x3_datasets = np.array([range(100), range(301,401),
                        range(77, 177), range(33, 133)]).T  # (100, 4)

y1 = np.array(range(3001, 3101))
                # 화성의 화씨 온도
y2 = np.array(range(13001, 13101)) # 비트코인가격

    
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets,
    x2_datasets,
    x3_datasets,
    y1,
    y2,
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

# #2-2. 모델
# input21 = Input(shape=(3,))
# dense21 = Dense(50, activation= 'relu', name='han21')(input21)
# dense22 = Dense(40, activation= 'relu', name='han22')(dense21)
# dense23 = Dense(30, activation= 'relu', name='han23')(dense22)
# dense24 = Dense(20, activation= 'relu', name='han24')(dense23)
# output21 = Dense(3, activation= 'relu', name='han25')(dense24)
# # model2 = Model(inputs = input21, outputs=output21)

# #2-3. 모델
# input31 = Input(shape=(4,))
# dense31 = Dense(50, activation= 'relu', name='han31')(input31)
# dense32 = Dense(40, activation= 'relu', name='han32')(dense31)
# dense33 = Dense(30, activation= 'relu', name='han33')(dense32)
# dense34 = Dense(20, activation= 'relu', name='han34')(dense33)
# output31 = Dense(10, activation= 'relu', name='han35')(dense34)

#2-4. 모델 합치기.
from keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)

# 2-5. 분기1.
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(last_dense2)

# 2-6. 분기2.
last_output2 = Dense(1, name='last2')(merge3)

model = Model(inputs=[input1], outputs = [last_output1, last_output2])
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    patience = 50,
    restore_best_weights=True,
)

model.fit([x1_train], [y1_train, y2_train], epochs=1000, batch_size=32,
          validation_split=0.1,
          callbacks=[es]
          )

#4. 평가, 예측
result = model.evaluate([x1_test], [y1_test, y2_test])
print('loss : ', result)
# loss :  [772.1495361328125, 699.477294921875, 72.67222595214844]

x1_pred = np.array([range(100, 106), range(400, 406)]).T



y_pred = model.predict([x1_pred])
print(y_pred)
# [array([[3057.3306],
#        [3060.4683],
#        [3062.6743],
#        [3063.269 ],
#        [3066.7136],
#        [3070.266 ]], dtype=float32), array([[13087.564],
#        [13098.949],
#        [13113.661],
#        [13117.592],
#        [13128.588],
#        [13139.974]], dtype=float32)]
# 예측값까지 완성
# 맹그러봐

