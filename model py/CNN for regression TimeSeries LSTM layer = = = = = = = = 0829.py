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


total = img2array('C:/Users/admin/Desktop/0828/camera cap data') #23550 데이터
#total2 = img2array('C:/Users/admin/Desktop/0823/camera cap data') #18950
#total3 = img2array('C:/Users/admin/Desktop/0821/camera cap data') #14014 = 56514

#total =  total1 + total2 + total3 

train = total[0:23000]
test = total[23000:]


#test = img2array('C:/Users/admin/Desktop/test   data원본/0812/camera cap data')



# 2. sol set 만들기

import pandas
import matplotlib.pyplot as plt


total_file1 = 'C:/Users/admin/Desktop/0828/angle_data_suit.csv'
#total_file2 = 'C:/Users/admin/Desktop/0823/angle_data_suit.csv'
#total_file3 = 'C:/Users/admin/Desktop/0821/angle_data_suit.csv'

#test_file = 'C:/Users/admin/Desktop/test   data원본/0812/angle_data_suit.csv'

names = ['num', 'pitch', 'yaw', 'roll']

df_total = pandas.read_csv(total_file1, names=names)
#df_total2 = pandas.read_csv(total_file2, names=names)
#df_total3 = pandas.read_csv(total_file3, names=names)

#df_total = df_total1.append(df_total2).append(df_total3)


#df_test = pandas.read_csv(test_file, names=names)

#a = list()
#a.append(df_total.yaw)
#a.append(df_train.pitch)
#b = list()
#b.append(df_test.yaw)
#b.append(df_test.pitch)

#Total에서 y_Train,y_Test 나누기
a = list()
a.append(df_total.yaw[0:23000])
a.append(df_total.yaw[23000:])
y_train = np.array(a[0]).T
y_test = np.array(a[1]).T


#y_test = np.array(b).T  # Transpose & array


y_train = y_train/91 # 1도 당 90.7 yaw transfom to angle
y_train = y_train[1:len(y_train)]
#y_train = y_train[0:7000]


y_test = y_test/91
y_test = y_test[1:len(y_test)]
#y_test = y_test[0:2000]

################################################# [t+1] - [t] 

def data_diff(dataset):
    train_diff , train = [], []
    for i in range(len(dataset)-1):
        diff = (dataset[i+1] - dataset[i])
        train_diff.append(diff)                            #tuple 형 len= 2 [i+1]-[i], i+1 데이터 array반환
        train.append(dataset[i+1])

    return np.array(train_diff), np.array(train)              #자동으로 0번째 (쓰레기데이터) 무시됨


x_train_diff = data_diff(train)[0]
x_train = data_diff(train)[1]
x_train_diff = x_train_diff.astype('float32')

# noise 처리
x_train_diff[x_train_diff<10] = 0
x_train_diff[x_train_diff>220] = 0

x_train = x_train.astype('float32')
x_train_stack = np.stack((x_train,x_train_diff),axis=3)   # (data num , 128,128, 2)


x_test_diff = data_diff(test)[0]
x_test = data_diff(test)[1]
x_test_diff = x_test_diff.astype('float32')
# noise 처리
x_test_diff[x_test_diff<10] = 0
x_test_diff[x_test_diff>220] = 0

x_test = x_test.astype('float32')
x_test_stack = np.stack((x_test,x_test_diff),axis=3)   # (data num , 128,128, 2)



################################## Seq화
x_train1 = []
seq_len = 2
for i in range(x_train_stack.shape[0]-1):
    x_train1.append(x_train_stack[i:i+seq_len,:,:])

x_train_stack = np.asarray(x_train1, dtype='float32')


x_test1 = []
seq_len = 2
for i in range(x_test_stack.shape[0]-1):
    x_test1.append(x_test_stack[i:i+seq_len,:,:])

x_test_stack = np.asarray(x_test1, dtype='float32')



# normalize
x_train_stack = x_train_stack / 255
x_test_stack = x_test_stack / 255



#seq화 할때 0번째 데이터 빠짐
y_train = y_train[1:len(y_train)]
y_test = y_test[1:len(y_test)]



# val train test split
x_train_stack_val = x_train_stack[23000:]
x_train_stack = x_train_stack[:23000]


y_train_val = y_train[23000:]
y_train = y_train[:23000]


print('x_train:',x_train_stack.shape)
print('x_train_val:',x_train_stack_val.shape)
print('x_test:' , x_test_stack.shape)

print('y_train:', y_train.shape)
print('y_train_val:',y_train_val.shape)
print('y_test:',y_test.shape)


#model 
import keras.backend.tensorflow_backend as K
import numpy as np
from numpy import array
from numpy import zeros
from random import random
from random import randint
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout,Flatten,merge, TimeDistributed
from sklearn.preprocessing import MinMaxScaler

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers

#x_train_stack = x_train_stack.reshape(x_train_stack.shape[0],10, 128,128,1).astype('float32')
#x_train_stack_val = x_train_stack_val.reshape(x_train_stack_val.shape[0],10, 128,128,1).astype('float32')
#x_test_stack = x_test_stack.reshape(x_test_stack.shape[0],10,128,128,1).astype('float32')

#x_train = x_train.reshape(x_train.shape[0],128,128,1).astype('float32')
#x_train_diff = x_train_diff.reshape(x_train_diff.shape[0],128,128,1).astype('float32')
#x_train_diff_val = x_train_diff_val.reshape(x_train_diff_val.shape[0],128,128,1).astype('float32')

#x_test = x_test.reshape(x_test.shape[0],128,128,1).astype('float32')
#x_test_diff = x_test_diff.reshape(x_test_diff.shape[0],128,128,1).astype('float32')


'''
# generate the next frame in the seq
def next_frame(last_setp, last_frame, column):
    #define the scope of the next setp
    lower = max(0,last_step-1)
    upper = min(last_frame.shape[0]-1, last_step+1)
    # choose the row index for the next step
    step = randint(lower, upper)
    #copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] =1
    return frame, step

# generate a seq of frames of a dot moving across an image
def build_frames(size):
    frames = list()
    #create the first frame
    frame = zeros((size,size))
    step = randint(0, size-1)
    # decide if we are heading left or right
    right = 1 if random() < 0.5 else 0
    col = 0 if right else size-1
    frame[step, col] = 1
    frames.append(frame)
    # create all remaining frames
    for i in range(1, size):
        col = i if right else size-1-i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)
    return frames, right

# generate multiple seq of frames and reshape for network input
def generate_examples(size, n_patterns):
    X, y = list(), list()
    for _ in range(n_patterns):
        frames, right = build_frames(size)
        X.append(frames)
        y.append(right)
    # resize as [samples, timesteps, width, height, channels]
    X = array(X).reshape(n_patterns, size, size, size, 1)
    y = array(y).reshape(n_patterns,1)
    return X,y
'''



#model 구축
with K.tf.device('/gpu:0'):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(None, 96, 128,2))))  # output (128,128) 32개
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  # output (64,64) 32개
    model.add(Dropout(0.2))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3 ), padding='same', activation='relu')))  # (64,64) 32개
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  # output (32,32) 32개
    

    model.add(TimeDistributed(Conv2D(32, (3, 3 ), padding='same', activation='relu')))  # (64,64) 32개
    
    model.add(TimeDistributed(Conv2D(32, (3, 3 ), padding='same', activation='relu')))  # (64,64) 32개
    
    
    
    
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(240,return_sequences=True))
    

    model.add(Flatten())
    model.add(Dropout(0.2))
    #model.add(Dense(40,init='normal',activation='relu'))
    model.add(Dense(1,init='normal',activation=None,))  # (yaw,pitch,roll) 예측

   
    

    
    #Compile model
    learning_rate = 0.09
    epochs = 100
    decay_rate = learning_rate / epochs

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping()  # 개선여지 없을때 조기종료
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2 = 0.999, epsilon=None, decay=0.0, amsgrad=False)

    
    model.compile(loss= 'mse', optimizer='adam')



    

    
    hist_256 = model.fit(x_train_stack ,y_train, validation_data=(x_train_stack_val,y_train_val),
                         batch_size=256, epochs=epochs,verbose=2 )



model.save('C:/Users/admin/Desktop/My_CNN+LSTM_model_0829.h5')

print('My_model.h5저장완료')


    

# 5. 학습과정 보기 batch 256

import matplotlib.pyplot as plt

plt.plot(hist_256.history['loss'])
plt.plot(hist_256.history['val_loss'])
plt.ylim(0.0, 200.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()





# 7. 모델 사용
yhat_test = model.predict(x_test_stack, batch_size= 128)


import matplotlib.pyplot as plt
import random
plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_test_stack)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[j].reshape(96,128))
    sub_plt.set_title('R yaw %.1f \n P yaw %.1f  '
                      % (y_test[j], yhat_test[j]))

plt.show()

# y_test, y_prediction 시각화
fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
#ax2=fig.add_subplot(2,1,2)

num = np.array(range(0,y_test.shape[0]))
num = num.reshape(num.shape[0],1)

ax1.plot(num,y_test[:], label ='real')
ax1.plot(num,yhat_test[:], label = 'predict')


#ax2.plot(num,y_test[:,1], label='real')
#ax2.plot(num,yhat_test[:,1], label='predict')
plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()




from keras.models import load_model
'''
model = load_model('C:/Users/admin/Desktop/My_model.h5')
'''






