# 1.input set

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


train0 = img2array('C:/Users/admin/Desktop/yolo_py/0820/camera cap data')
train1 = img2array('C:/Users/admin/Desktop/yolo_py/0816/camera cap data')
train2 = img2array('C:/Users/admin/Desktop/yolo_py/0812/camera cap data')



#test1 = img2array('C:/Users/admin/Desktop/test   data원본/0812/camera cap data')
#test2 = img2array('C:/Users/admin/Desktop/test   data원본/0808/camera cap data')



train = train0 + train1 + train2

x_train = img_train = np.array(train)  # nd array for test

x_train = x_train.astype('float32')




################################## 2. sol set 만들기

import pandas
import matplotlib.pyplot as plt

train0_file = 'C:/Users/admin/Desktop/yolo_py/0820/angle_data_suit.csv'
train1_file = 'C:/Users/admin/Desktop/yolo_py/0816/angle_data_suit.csv'
train2_file = 'C:/Users/admin/Desktop/yolo_py/0812/angle_data_suit.csv'





names = ['num', 'pitch', 'yaw', 'roll']


df_train0 = pandas.read_csv(train0_file, names=names)
df_train1 = pandas.read_csv(train1_file, names=names)
df_train2 = pandas.read_csv(train2_file, names=names)

df_train = df_train0.append(df_train1).append(df_train2)


a = list()
#a.append(df_train.pitch)
a.append(df_train.yaw)



y_train = np.array(a).T




y_train = y_train/91 # 1도 당 90.7 yaw transfom to angle
y_train = y_train[1:len(y_train)]



import cv2

def data_diff(dataset):
    train_diff, train = [], []
    for i in range(len(dataset)-1):
        diff= cv2.subtract(dataset[i+1],dataset[i])
        train_diff.append(diff)
        train.append(dataset[i+1])                            #tuple 형 len= 2 [i+1]-[i], i+1 데이터 array반환
    return np.array(train_diff), np.array(train)              #자동으로 0번째 (쓰레기데이터) 무시됨



x_train_diff = data_diff(x_train)[0]
x_train = data_diff(x_train)[1]

x_train_diff = x_train_diff.astype('float32')
x_train = x_train.astype('float32')

x_train_stack = np.stack((x_train,x_train_diff),axis=3)   #원래 data + diff -1까지 합침






# normalize
x_train_stack = x_train_stack / 255


# val train split


x_train_stack_val = x_train_stack[22000:]
x_train_stack = x_train_stack[:22000]


y_train_val = y_train[22000:]
y_train = y_train[:22000]


print(x_train_stack.shape)
print(y_train.shape)
print(x_train_stack_val.shape)
print(y_train_val.shape)

#model 


import keras.backend.tensorflow_backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers


#x_train = x_train.reshape(x_train.shape[0],128,128,1).astype('float32')
#x_train_diff = x_train_diff.reshape(x_train_diff.shape[0],128,128,1).astype('float32')
#x_train_diff_val = x_train_diff_val.reshape(x_train_diff_val.shape[0],128,128,1).astype('float32')

#x_test = x_test.reshape(x_test.shape[0],128,128,1).astype('float32')
#x_test_diff = x_test_diff.reshape(x_test_diff.shape[0],128,128,1).astype('float32')



## dataset 시각화
import matplotlib.pyplot as plt
import random


plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10,10)

    
f, axarr = plt.subplots(plt_row, plt_col)



for i in range(plt_row*plt_col):

    j=random.randrange(0,len(x_train_stack)-1)
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
                 input_shape=(128, 128,2)))  # output (128,128) 32개
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
    epochs = 30
    decay_rate = learning_rate / epochs

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping()  # 개선여지 없을때 조기종료
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2 = 0.999, epsilon=None, decay=0.0, amsgrad=False)

    
    model.compile(loss= 'mse', optimizer='adam')
    hist_256 = model.fit(x_train_stack, y_train, validation_data=(x_train_stack_val,y_train_val),
                         batch_size=512, epochs=epochs,verbose=2 )



model.save('C:/Users/admin/Desktop/model_yolo.h5')



    

# 5. 학습과정 보기 batch 256

import matplotlib.pyplot as plt

plt.plot(hist_256.history['loss'])
plt.plot(hist_256.history['val_loss'])
plt.ylim(0.0, 500.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# 7. 모델 사용
yhat_train_val = model.predict(x_train_stack_val, batch_size= 128)

import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_train_stack_val)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_train_stack_val[j].T[1].T)
    sub_plt.set_title('Real yaw %.1f \n Predict yaw %.1f '
                      % (y_train_val[j], yhat_train_val[j]))


plt.show()

f, axarr = plt.subplots(plt_row, plt_col)
for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_train_stack_val)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_train_stack_val[j].T[0].T)
    sub_plt.set_title('Real yaw %.1f \n Predict yaw %.1f '
                      % (y_train_val[j], yhat_train_val[j]))
    
plt.show()
    
# y_test, y_prediction 시각화
num = np.array(range(0,y_train_val.shape[0]))
num = num.reshape(num.shape[0],1)
plt.plot(num,y_train_val, label ='real')
plt.plot(num,yhat_train_val, label = 'predict')

plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()








'''


#######################모델 테스트 ######################
#######################모델 테스트 ######################
#######################모델 테스트 ######################
#######################모델 테스트 ######################

train0 = img2array('C:/Users/admin/Desktop/yolo_py/0820/camera cap data')
train1 = img2array('C:/Users/admin/Desktop/yolo_py/0816/camera cap data')
train2 = img2array('C:/Users/admin/Desktop/yolo_py/0812/camera cap data')

test = train0 + train1 + train2 #41743

x_test = img_test = np.array(test)

x_test = x_test.astype('float32')


train0_file = 'C:/Users/admin/Desktop/yolo_py/0820/angle_data_suit.csv'
train1_file = 'C:/Users/admin/Desktop/yolo_py/0816/angle_data_suit.csv'
train2_file = 'C:/Users/admin/Desktop/yolo_py/0812/angle_data_suit.csv'


df_train0 = pandas.read_csv(train0_file, names=names)
df_train1 = pandas.read_csv(train1_file, names=names)
df_train2 = pandas.read_csv(train2_file, names=names)



df_test = df_train0.append(df_train1).append(df_train2)


t = list()
#a.append(df_train.pitch)
t.append(df_train.yaw)

y_test = np.array(t).T




y_test = y_test/91 # 1도 당 90.7 yaw transfom to angle
y_test = y_test[1:len(y_test)]



x_test_diff = data_diff(x_test)[0]
x_test = data_diff(x_test)[1]

x_test_diff = x_test_diff.astype('float32')
x_test = x_test.astype('float32')

x_test_stack = np.stack((x_test,x_test_diff),axis=3)

x_test_stack = x_test_stack / 255







yhat_test = model.predict(x_test_stack, batch_size= 128)


import matplotlib.pyplot as plt

plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_test_stack)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test_stack[j].T[1].T)
    sub_plt.set_title('Real yaw %.1f \n Predict yaw %.1f '
                      % (y_test[j], yhat_test[j]))



plt.show()

f, axarr = plt.subplots(plt_row, plt_col)
for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_test_stack)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test_stack[j].T[0].T)
    sub_plt.set_title('Real yaw %.1f \n Predict yaw %.1f '
                      % (y_test[j], yhat_test[j]))
    
plt.show()
    
# y_test, y_prediction 시각화
num = np.array(range(0,y_test.shape[0]))
num = num.reshape(num.shape[0],1)
plt.plot(num,y_test, label ='real')
plt.plot(num,yhat_test, label = 'predict')

plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()




'''



