# 9-1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 4,
            validation_data = (x_val, y_val),

          verbose = 1,
          ) # 배치 사이즈가 전체 데이터보다 클 경우 통배치가 돌아간다.
# verbose = 0 : 침묵
# verbose = 1 : 디폴트
# verbose = 2 : 프로그레스바 삭제
# verbose = 나머지 : 에폭만 나옴
# 성능에 영향을 미치진 않는다.


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 통상적으로 평가 loss는 훈련 loss보다 높게 나온다. 훈련 데이터에 과적합이 일어나면 평가 loss가 더 높게 나온다.
# train과 달리 최적의 weight를 업데이트하지 않기 때문에 테스트셋 evaluate는 배치를 통으로 돌린다.
print('loss : ', loss)