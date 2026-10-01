import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밋네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5,
# '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10,
# '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16,
# '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21,
# '재미없다': 22, '재밋네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26,
# '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)
#[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14],
# [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23],
# [24, 25], [26, 27], [28, 29, 30]]

############# 패딩 ##################
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding='pre', # 뒤에 post
                         maxlen=5, 
                         truncating='post' # 디폴트 앞이 짤렷다.
                         )

#2. 모델
from keras.layers import Embedding
model = Sequential()
#################### 임베딩 1 ####################

# model.add(Embedding(input_dim=30, output_dim=100, input_length=5))
#                     #단어사전의 갯수, 차원
# model.add(SimpleRNN(10))
# model.add(Dense(1))
# model.summary()

# embedding (Embedding)       (None, 5, 100)            3000      
                                                                 
# simple_rnn (SimpleRNN)      (None, 10)                1110      

# exit()

#################### 임베딩 2 ####################
# model.add(Embedding(input_dim=30, output_dim=100)) # input_length 명시안해도 알아서 맞춰줘
#                     #단어사전의 갯수, 차원
# model.add(SimpleRNN(10))
# model.add(Dense(1))

#################### 임베딩 3 ####################
# model.add(Embedding(30, 100)) # input_dim, output_dim
# model.add(Embedding(30, 100, 5)) # input_dim, output_dim, 안된다!!!
model.add(Embedding(30, 100, input_length=5)) 
                    #단어사전의 갯수, 차원
model.add(SimpleRNN(10))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
            optimizer ='adam',
            metrics=['acc'])

model.fit(padded_x, labels, epochs=100, batch_size=4)
