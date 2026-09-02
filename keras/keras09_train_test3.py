import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3 나눈다
# 힌트 : 사이킷런
# 정확한 판단을 위해 작업한다.
x_train, x_test, y_train, y_test = train_test_split(
    x, y,   # 데이터
    train_size=0.7, # 훈련셋 70%
    # test_size=0.3,  
    # shuffle=True,   # 디폴트 섞는다.
    random_state=333,  # 랜덤 시드 고정
)
print('x_train:', x_train)
print('x_test:', x_test)
print('y_train:', y_train)
print('y_test:', y_test)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 4)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
