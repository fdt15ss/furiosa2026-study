# 토큰 - 음절이나 어절 단위로 잘라놓은 것.

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 못생겼다. 길동이는 마구 마구 더 잘생겼다.'

token = Tokenizer() # 토크나이저 클래스를 정의한다. # 인스턴스(객체) = 클래스(), 인스턴스 생성
token.fit_on_texts([text, text2])

print(token.word_index)

# 맹글기
x = token.texts_to_sequences([text, text2])
print(x)


encoder = OneHotEncoder(sparse_output=False)
print(x[0])
print([np.array(x[0]), np.array(x[1])])
x = np.concatenate([np.array(x[0]), np.array(x[1])])
print(x)
x = encoder.fit_transform(np.array(x).reshape(-1,1))
print(x)
print(type(x)) # <class 'numpy.ndarray'>
print(x.shape) # (24, 17)