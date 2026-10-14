import warnings
warnings.filterwarnings("ignore")

from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet201, DenseNet169, DenseNet121
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetMobile, NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications import Xception

model_list = [VGG16(), VGG19(), ResNet50(), ResNet50V2(), ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
              DenseNet201(), DenseNet169(), DenseNet121(), InceptionV3(), InceptionResNetV2(),
              MobileNet(), MobileNetV2(), MobileNetV3Small(), MobileNetV3Large(), NASNetMobile(), NASNetLarge(),
              EfficientNetB0(), EfficientNetB1(), EfficientNetB2()]

model = Xception()
model.trainable = True

model.summary()

print("===============================")
print("모델명 : ", model.name)
print("전체 가중치 갯수 : ", len(model.weights))
print("전체 훈련가능 가중치 갯수 : ", len(model.trainable_weights))

for model in model_list:
    # model.trainable = False
    print("===============================")
    print("모델명 : ", model.name)
    print("전체 가중치 갯수 : ", len(model.weights))
    print("전체 훈련가능 가중치 갯수 : ", len(model.trainable_weights))