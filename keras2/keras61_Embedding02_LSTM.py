import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘 만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘': 5,
# '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
# '한': 11, '번': 12, '더': 13, '보고': 14, '싶어요': 15, '글쎄': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19,
# '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, 
# '재밋네요': 24, '개똥이': 25, '바보': 26, '말똥이': 27,
# '잘생겼다': 28, '길동이': 29, '또': 30, '구라친다': 31}

x = token.texts_to_sequences(docs)
print(x)
#[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15],
# [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24],
# [25, 26], [27, 28], [29, 30, 31]]

############# 패딩 ##################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding='pre', # 뒤에 post
                         maxlen=5, 
                         truncating='post' # 디폴트 앞이 짤렷다.
                         )
padded_x = padded_x.reshape(padded_x.shape[0], padded_x.shape[1], 1)
print(padded_x)
print(padded_x.shape) # (15, 5, 1)

train_x, test_x, train_y, test_y = train_test_split(padded_x, labels, train_size=0.66, random_state=42)
# exit()

model = Sequential()
model.add(LSTM(32, input_shape=(5, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) # 이진분류

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',
                   patience=50,
                   mode='min',
                   restore_best_weights=True,
                   )

start_time = time.time()
model.fit(train_x, train_y, epochs=1000,
          batch_size=2,
          validation_split=0.1,
          callbacks=[es],
          )

train_time = time.time() - start_time

print(f"Training time: {train_time} seconds")

# 4. 평가, 예측
loss, acc = model.evaluate(test_x, test_y)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {acc}")

x_predict = ['개똥이 잘생겼다']
# x_predict = ['개똥이 구라친다']
x_predict_tokenized = token.texts_to_sequences(x_predict)
x_predict_padded = pad_sequences(x_predict_tokenized,
                                 padding='pre', maxlen=5,
                                 truncating='post')
x_predict_padded = x_predict_padded.reshape(x_predict_padded.shape[0], x_predict_padded.shape[1], 1)
y_predict = model.predict(x_predict_padded)
print('y_predict :', y_predict)