# 33 - 1 카피
# R2 기준 0.55

from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import my_util


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


# exit()

x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.5, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.5, random_state=42)


# minmaxscaler
"""
MinMaxScaler

원값 - Min
-----------
Max - Min
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() # 최소값과 최대값을 기준으로 변환
# scaler = StandardScaler() # 정규화
# scaler = MaxAbsScaler() # 최대절대값으로 나눠줌
scaler = RobustScaler() # RobustScaler는 이상치에 덜 민감하다. 이상치에 강하다. IQR 사용


# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) #fit과 transform이 한 번에 가능하다.

x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
print(x_train)
print(np.min(x_train),np.max(x_train))  # 0.0 1.0000000000000004
print(np.min(x_val),np.max(x_val))      # -0.005005005005005003 1.3337231968810916
print(np.min(x_test),np.max(x_test))    # -0.0010638297872338498 1.0

# exit()

#2. 모델 구성
# print(x_train.shape)

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dropout(0.2))
model.add(Dense(12))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dropout(0.5))
model.add(Dense(1))

# model.summary()

path = './_save/keras33/'
# model.save(path + 'keras29_1_save_model.keras')
# model.save_weights(path + 'keras29_5_save_1.weights.h5') #초기 ㅏ중치
# model = load_model(path + 'keras29_1_save_model.keras') # 초기 가중치 상태
# model.summary()

# exit()


#3. 컴파일, 훈련
from keras.optimizers import Adam
# learning_rate = 0.01
# learning_rate = 0.001     # 디폴트
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss = 'mse', optimizer =Adam(learning_rate=learning_rate))
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    save_best_only=True, 
    filepath=path + 'keras52_mcp1.keras'
)

BATCH_SIZE = 32

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000,
                 batch_size=BATCH_SIZE,
                 validation_data=(x_val, y_val), callbacks=[es, mcp],
                 verbose=1)
end_time = time.time()

# model.save(path + 'keras29_3_save_model.keras')
# model.save_weights(path + 'keras29_5_save_2.weights.h5')
# model.load_weights(path + 'keras29_5_save_2.weights.h5')


#4. 평가, 예측
train_time = round(end_time - start_time, 3)
print('총 시간 :',train_time, '초')
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# print("========================== hist =========================")
# print(hist)
# print("========================== hist.history =========================")
# print(hist.history)
# print("========================== loss =========================")
# print(hist.history['loss'])
# print("========================== val_loss =========================")
# print(hist.history['val_loss'])
# print("=============================================================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False
# plt.figure(figsize=(9, 6))
# plt.plot(hist.history['loss'][2:], c='red', label='loss') # y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
# plt.title('캘리포니아 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')   # 우측 상단에 라벨 표시
# plt.grid() # 격자표시 추가
# plt.show()

my_util.record_model_csv(
    model = model,
    data_shape = x_train.shape,
    random_num = 0,
    batch_size = BATCH_SIZE,
    # batch_size = 8,
    # history = None,
    history = hist,
    training_time = train_time,
    # training_time = 2712.563,
    test_loss = loss,
    sub_score = r2,
    train_ration = 0,
    csv_file_path="01_california.csv"
)

# loss : 0.6413015723228455
# r2(16-8-1) :  0.5116739162297916
# r2 :  0.5006368636230876
# r2 :  0.4866390989900473
# r2(16-1) :  0.010727020275410504
# r2(16-12-8-1) :  0.5249744806146608
# r2(x 통으로 MinMaxScaler 적용) :  0.5977489562972026
# r2(스케일러 train셋에만 적용) :  0.5965634490605173
# r2(StandardScaler 적용) :  0.5981213236492424
# r2(MaxAbsScaler 적용) :  0.5537398859519264
# r2(RobustScaler 적용) :  0.5972836194750686
# loss : 0.5253134965896606
# r2 :  0.5966847309735741

# loss : 0.7992951273918152
# r2 :  0.38633238601293285

# loss : 0.8004339933395386
# r2 :  0.38545809907706263

# loss(dropout) : 0.5299391746520996
# r2(dropout) :  0.5931335288987801

# loss(optimizer-adam lr 0.01) : 0.6259700655937195
# r2(optimizer-adam lr 0.01) :  0.5194047648626794

# loss : 0.7423447966575623
# r2(lr 0.001) :  0.4300564874703119

# loss : 521.4085083007812
# r2(lr 0.05) :  -399.31706809236834

# loss : 0.5565659403800964
# r2 :  0.5726903987522372