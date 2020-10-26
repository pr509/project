from bing_image_downloader import downloader
downloader.download('narendra modi image',limit=10,output_dir='primeminister',adult_filter_off=True,force_replace=False,timeout=60)
downloader.download('donald trump image',limit=10,output_dir='primeminister',adult_filter_off=True,force_replace=False,timeout=60)
downloader.download('imran khan image',limit=10,output_dir='primeminister',adult_filter_off=True,force_replace=False,timeout=60)
downloader.download('boris johnson image',limit=10,output_dir='primeminister',adult_filter_off=True,force_replace=False,timeout=60)
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
import os
target =[]
images=[]
flat_data =[]
DATADIR='primeminister'
CATEGORIES=['narendra modi image','donald trump image','imran khan image','boris johnson image']
for category in CATEGORIES:
    class_num=CATEGORIES.index(category)
    #print(class_num)
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)
target=np.array(target)
images=np.array(images)
flat_data=np.array(flat_data)
var1=input('enter your image :')
var1_array=imread(var1)
var1_resized=resize(var1_array,(150,150,3))
flat_data.append(var1_resized.flatten())
images.append(img_resized)
df=pd.DataFrame(flat_data)
df['target']=target
#print(df)
x=df.iloc[:,0:67500].values
y=df.iloc[:,67500].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=6,metric='euclidean')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))
ynew_pred=model.predict()


