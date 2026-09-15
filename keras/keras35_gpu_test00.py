# python=3.10
# conda 환경명 : tf29x-gpu
# numpy==1.26.4
# scikit-learn==1.7.2
# pandas==2.3.3
# pip install matplotlib==3.10.9

import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
if(gpus):
    print('GPU 있다!~!')
else:
    print('GPU 없다!!!')
