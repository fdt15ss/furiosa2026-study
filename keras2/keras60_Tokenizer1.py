# 토큰 - 음절이나 어절 단위로 잘라놓은 것.

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer() # 토크나이저 클래스를 정의한다. # 인스턴스(객체) = 클래스(), 인스턴스 생성
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)
# [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
print(len(x))

# 원핫 인코딩 3가지 만들기

# import pandas as pd
#1. pandas
# x = pd.get_dummies(np.array(x).reshape(-1), dtype=int)
# print(x)
# print(x.shape) # (14, 9)

#2. sklearn
# encoder = OneHotEncoder(sparse_output=False)
# # x = np.array(x)

# x = encoder.fit_transform(np.array(x).reshape(-1,1))
# print(x)
# print(type(x)) # <class 'numpy.ndarray'>
# print(x.shape) # (14, 9)


#3. keras
x = to_categorical(np.array(x).reshape(-1) - 1)
print(x)
print(x.shape) # (14, 9)
