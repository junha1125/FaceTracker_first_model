# 1.input data 준비.

# image to array transform for training
import os
import numpy as np
from PIL import Image

def img2array(imagepath):
    img_arr_list = list()
    for (path, dirname, files) in os.walk('%s' % imagepath):
        for f in files:
            os.chdir('%s' % imagepath)
            img = Image.open(f)
            arr = np.array(img)
            img_arr_list.append(arr)  # list화
    return img_arr_list

train0 = img2array('C:/Users/kmh03/OneDrive/바탕 화면/newproj/gg')
train = train0  #33407
x_train = img_train = np.array(train)  # nd array for test
x_train = x_train.astype('float32')

# image data normalize
x_train = x_train / 255

# 2. sol set 만들기
import pandas
import matplotlib.pyplot as plt

train0_file = 'C:/Users/kmh03/OneDrive/바탕 화면/newproj/angle_data_suit.csv'
names = ['num', 'pitch', 'yaw', 'roll']
df_train0 = pandas.read_csv(train0_file, names=names)
df_train = df_train0
a = list()
a.append(df_train.yaw)
#a.append(df_train.pitch) # pitch 도 추가하고 싶다면 할 수 있다.
y_train = np.array(a).T  # Transpose & array
y_train = y_train/91 # 1도 당 90.7 yaw transfom to angle
print(x_train.shape)
print(y_train.shape)


import keras.backend.tensorflow_backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
x_train = x_train.reshape(x_train.shape[0],128,128,1).astype('float32')

## dataset 시각화
import matplotlib.pyplot as plt
import random


plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10,10)

    
f, axarr = plt.subplots(plt_row, plt_col)



for i in range(plt_row*plt_col):

    j=random.randrange(0,len(x_train)-1)
    sub_plt = axarr[i//plt_row, i%plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_train[j].reshape(128,128))
    sub_plt.set_title('Real yaw %.1f '
                      % (y_train[j])) 


plt.show()






#model 구축
with K.tf.device('/gpu:0'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(128, 128,1)))  # output (128,128) 32개
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))  # output (64,64) 32개
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))  # (64,64) 32개
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))  # output (32,32) 32개
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))  # (64,64) 32개
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))  # (64,64) 32개
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(128, init='normal' , activation='relu'))
    model.add(Dense(64, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1,init='normal',activation=None,))  # (yaw,pitch,roll) 예측

    #Compile model
    learning_rate = 0.05
    epochs = 10
    decay_rate = learning_rate / epochs

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping()  # 개선여지 없을때 조기종료
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2 = 0.999, epsilon=None, decay=0.0, amsgrad=False)

    
    model.compile(loss= 'mse', optimizer='adam')
    hist_256 = model.fit(x_train, y_train, validation_split=0.15, shuffle=True,
                         batch_size=256, epochs=epochs,verbose=2 )
    #hist_256 = model.fit(x_train, y_train, validation_split=0.2, shuffle= True , batch_size=512, epochs=epochs,verbose=2 )


model.save('C:/Users/admin/Desktop/인수인계/model_yolo_CNN.h5')
    

# 5. 학습과정 보기 batch 256

import matplotlib.pyplot as plt

plt.plot(hist_256.history['loss'])
plt.plot(hist_256.history['val_loss'])
plt.ylim(0.0, 1000.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# 7. 모델 사용
yhat_test = model.predict(x_train, batch_size= 128)

import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_train)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_train[j].reshape(128,128))
    sub_plt.set_title('Real yaw %.1f \n Predict yaw %.1f '
                      % (y_train[j], yhat_test[j]))


plt.show()

# y_test, y_prediction 시각화
num = np.array(range(0,y_train.shape[0]))
num = num.reshape(num.shape[0],1)
plt.plot(num,y_train, label ='real')
plt.plot(num,yhat_test, label = 'predict')

plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()





