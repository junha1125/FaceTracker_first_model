import os
import numpy as np
from PIL import Image




   
def img_size128_train(imagepath):
    print('변환할 image pixels width 입력:')
    w = int(input())
    print('변환할 image pixels height 입력:')
    h = int(input())
    size = w, h
    for (path, dirname, files) in os.walk('%s' % imagepath):
        for f in files:
            os.chdir('%s' % imagepath)
            im = Image.open(f)
            #im = im.crop((210,120,430,340)) #centered & resize rectangle 640x480일때
            im = im.convert('L') # 8bit grayscale
            im.thumbnail(size)
            im.save(f)
    return print('img size가 ',im.size,'로 저장되었습니다.')

def img_size128_test(imagepath):
    size = 128, 128
    for (path, dirname, files) in os.walk('%s' % imagepath):
        for f in files:
            os.chdir('%s' % imagepath)
            im = Image.open(f)
            #im = im.crop((210,120,430,340)) #centered & resize rectangle 640x480일때
            im = im.convert('L') # 8bit grayscale
            im.thumbnail(size)
            im.save(f)
    return print('img size가 ',im.size,'로 저장되었습니다.')

'''
key = input('image file 변환 하시겠습니까?(y/n)')
if key == 'y':
    print('image path를 입력하세요(fullpath):')
    imagepath = str(input())
    img_size128_train(imagepath) # 이미지 크기 128로 조정후 그 경로에 다시저장
else:
    pass
'''


#img_size128_test('C:/Users/admin/Desktop/0821/camera cap data')



##################################이미지 전처리 끝######################

########IR train data csv파일 만들기







import pandas
import matplotlib.pyplot as plt


filename1 = str(input('trackir data path 입력:'))

names1 = ['pitch','yaw','roll','x','y','z','stats','frame','measuretime','Time']
df_ir = pandas.read_csv(filename1, names=names1)


filename2 = str(input('camera time path 입력:'))
names2 = ['camera time']
df_cam = pandas.read_csv(filename2, names=names2)


array_cam = df_cam.values  # 데이터프레임에서 값만 array로 받아옴
array_ir = df_ir.values

array_ir_Time = df_ir['Time'].values

j = 0;
approximate_list = list()

for i in range(len(df_cam)):
    while (df_ir['Time'][j] < df_cam['camera time'][i]):
        j += 1
       
    slope = (array_ir[j, :] - array_ir[j - 1, :]) / (array_ir[j,9] - array_ir[j-1,9])
   
    interpolation = array_ir[j-1,:] + slope*(df_cam['camera time'][i]-array_ir[j-1,9])
    
    approximate_list.append(interpolation)

    
    if (j == len(df_ir)-1):
        break


save1 = 'data_suit.csv'
save2 = 'angle_data_suit.csv'

approximate_df = pandas.DataFrame(approximate_list,columns=names1)

approximate_df.to_csv(save1 ,header=False)

angle_df = approximate_df[['pitch','yaw','roll']]
angle_df.to_csv(save2 ,header=False)


