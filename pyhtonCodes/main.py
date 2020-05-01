# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:51:16 2020

@author: Erdinc
"""

#%%Libaries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import random
from keras import backend as K

#%%
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#%%datasetin projeye dahil edilmesi ve bilgilerinin gösterilmesi -- Veri Manipülasyonu Bölümü
def readDatasetNinformantions():    
    product=pd.read_excel("datasets/product.xlsx")
    product.head()
    df = product.copy()
    df.head()
    #data information
    df.info()
    df.dtypes
    #Numrical details
    df.describe().T
#    return df

#def setAreaColumnType():
    #df=readDatasetNinformantions()
    df['EkilenAlan'] = pd.to_numeric(df['EkilenAlan'], errors='coerce')
    return df

#%%Verilerin boş değerlerini tespit etmemiz gereken alan -- Veri Analizi Bölümü
def checkEmptyValues():
    df = readDatasetNinformantions()
    print("Is any value empty ? \n\n\n")
    if df.isnull().values.any() == True:
         df=df.dropna()
         return df
    else:
        return df

#%%datasetimizin özeti belirleyebilmek için bir fonksiyon yazacağız
def summaryofDataset():
    #df=checkEmptyValues()
    print("Üretim : Avg : "+str(df["Üretim"].mean()))
    print("Tohum Fiyatı : Avg : "+str(df["Tohum Fiyatı"].mean()))
    print("SatışFiyatı : Avg : "+str(df["SatışFiyatı"].mean()))
    print("Mazot : Avg : "+str(df["Mazot"].mean()))
    print("Gübre : Avg : "+str(df["Gübre"].mean()))
    print("Aylara göre ekim yapılma dağılımları : "+str(df["EkildiğiAy"].value_counts())) 
    #datasetimizi  grafiklerimizi oluşuturuyoruz
    sns.scatterplot(x="Üretim" , y= "Bölge", hue="Tohum Fiyatı",size="Tohum Fiyatı" ,data=df);
#def crossingWithScatterplot():
#    summaryofDataset()  
#crossingWithScatterplot()

#%%Örnek Teorisi - Veri Bilimi ve İstatislik Bölümü 
#Rastgele seçilen bir sütunda nasıl bir ortalama çıkaracağı sonucunu öğrenmek için 
#Örneklem teorisine başvurduk.
def theoryofExampleforSeedPrice():
#    df=checkEmptyValues()
    np.random.seed(10) 
    example = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    example2 = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    example3 = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    example4 = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    example5 = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    example6 = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    example7 = np.random.choice(a=df["Tohum Fiyatı"],size=50)
    newValue = (example.mean()+example2.mean()+example3.mean()+example4.mean()+example5.mean()+example6.mean()+example7.mean())/6
    print(newValue)
    print(df.mean())
    return newValue

#%%Betimsel İstatistik Uygulamaları
import researchpy as rp
def descriptiveStatistics():
    df=checkEmptyValues()
    rp.summary_cont(df[["Üretim","Tohum Fiyatı"]])
    rp.summary_cat(df[["EkildiğiAy","Bölge","ÜretimSüresi"]])
    df[["Üretim","EkilenAlan"]].cov()

#%%Bernoulli Kavramını uygulama
from scipy.stats import bernoulli
def bernolliCase():
    p= df["Tohum Fiyatı"].min()
    #p = 0.6 
    r = bernoulli(p)
    r.pmf(k=0)  
    return r

#%%Karar Aralığı
import statsmodels.stats.api as sms
def believeCase():
    df=checkEmptyValues()
    sms.DescrStatsW(df["Üretim"]).tconfint_mean()
    df["Üretim"].mean()

#%%Binom Kavramı
from scipy.stats import binom
def binomCase():
    p=0.01
    n=100
    rv=binom(n,p)
    rv.pmf(5)
    return rv

#%%Possion Kavramı 
from scipy.stats import poisson
def possionCase():
    lambda_ =0.1
    rv=poisson(mu = lambda_)
    rv.pmf(k=0)
    return rv

#%%Veri Önişleme - Aykırı Gözlem
def preProc():
    df=checkEmptyValues()
    dataTable = df["EkilenAlan"].copy()
    sns.boxplot(x = dataTable)
    q1 = dataTable.quantile(0.25)
    q3 = dataTable.quantile(0.75)
    iqr =q3-q1
    
    subLine = q1 - 1.5*iqr
    subLine 
    topLine = q3 + 1.5*iqr
    topLine 
    
    (dataTable <(subLine)) | (dataTable > (topLine))
    (dataTable <(subLine))
    cObservation = dataTable<subLine
    cObservation.head(10)
    contradictory=dataTable[cObservation]
    contradictory.index
    #contradictory is empty.i dont have any contradictory
    dataTable = df["Üretim"].copy()
    sns.boxplot(x = dataTable)
    q1 = dataTable.quantile(0.25)
    q3 = dataTable.quantile(0.75)
    iqr =q3-q1
    
    subLine = q1 - 1.5*iqr
    subLine 
    topLine = q3 + 1.5*iqr
    topLine 
    
    (dataTable <(subLine)) | (dataTable > (topLine))
    (dataTable <(subLine))
    cObservation = dataTable<subLine
    cObservation.head(10)
    contradictory=dataTable[cObservation]
    contradictory.index
    #contradictory is empty.i dont have any contradictory
    return dataTable

#%%Çok değişkenli aykırı gözlem
from sklearn.neighbors import LocalOutlierFactor
def preProc2():
    dataTable=preProc()
    X = np.r_[dataTable]
    
    LOF = LocalOutlierFactor(n_neighbors = 20 , contamination = 0.1)
    LOF.fit_predict(X)
    X_score = LOF.negative_outlier_factor_
    return X_score

#%%empty area for category veriable 
def checkEmptyAreaAgain():
    df=checkEmptyValues()
    df.isnull()
    df.groupby("EkildiğiAy")["EkildiğiAy"].count()
    df.groupby("Bölge")["Bölge"].count()
    df.groupby("ÜretimSüresi")["ÜretimSüresi"].count()
    return df

#%%Koalasyon grafiği
def cGraph():
    df=checkEmptyValues()
    listFeature = ["Ürün", "Üretim", "Tohum Fiyatı",'SatışFiyatı', 'Mazot', 'Gübre', 'EkilenAlan']
    sns.heatmap(df[listFeature].corr(), annot = False , fmt = ".2f")
    plt.show()

#%%Kar zarar durum Feature umuz için değişkenllerimizin son düzenlemelerini yağacağız
#Index(['Ürün', 'Üretim', 'EkildiğiAy', 'Tohum Fiyatı', 'Bölge', 'ÜretimSüresi','SatışFiyatı', 'Mazot', 'Gübre', 'EkilenAlan']       

def createFeature():
    df=checkEmptyValues()   
    
    #M2,Region,PTime,Mounth,Felitizer=getInputs()
    M2=1
    #Giderler
    #Tohum Fiyatı,Mazot,Gübre
    #Üretim Oranı
    #m2 başına ton bazında üretim    
    #Tohum Fiyatı
    #Genellikle hektar başına ortalama 100 – 160 kg tohum gerekir
    randomSeedNumber = random.randint(100, 160)
    df["tohumGider"]=M2*df["Tohum Fiyatı"]*randomSeedNumber
    
    #Mazot Fiyatı
    #1 Dönüm tarlaya 50lt mazot harcanıyor 
    #randomDieselNumber = random.uniform(50, 60)
    df["mazotGider"]= df["Mazot"]*M2*10
    df["mazotGider"][df.ÜretimSüresi=="Uzun Süre "]=df["Mazot"][df.ÜretimSüresi=="Uzun Süre "]*M2*20
    
    #Gübre Fiyatı
    #Uzun Süreli ürünlere 70-80kg ,kısa süreli ürünlere 20-40kg gübre atılır
    #1 Gübre torbasına(50kg) göre fiyatlandırılmış sütunumuzu seçicez 
    #1 dönüüm tarlaya 2 torba gübre gider 
    #randomFelitizerNumberforLongT = random.randint(70, 80)
    #randomFelitizerNumberforShortT = random.randint(20, 40)
    
    df["gübreGider"]= df["Gübre"]*M2*2*2
    df["gübreGider"][df.ÜretimSüresi=="Uzun Süre "]=df["Gübre"][df.ÜretimSüresi=="Uzun Süre "]*M2*2*2
    
    #↑Toplam Gideri Hesaplayalım
    df["toplamGider"]=df["tohumGider"]+df["mazotGider"]+df["gübreGider"]
    
    #Gelirler
    #Satış Fiyatı
    df["satışGelir"]=df["SatışFiyatı"]*(M2*100)
    
    #Gübre kullanımı verimi %20 arttırır.
    #Gübre kullanımının getireceği %20lik verim
    df["gübreVerimi"]=(df["satışGelir"]*2)%20
    df["gübreVerimi"][df.ÜretimSüresi=="Kısa Süre"]=(df["satışGelir"][df.ÜretimSüresi=="Kısa Süre"]*M2*2)%20
    
    #Çiftçi Destek Parası
    df["destekParasıGelir"]=df["DestekParası"]*M2*10
    
    
    #Toplam Gelir
    df["toplamGelir"]= df["destekParasıGelir"]+df["satışGelir"]+df["gübreVerimi"]
    
    #Kar - Zarar Durumu 
    df["sonuç"]=(df["toplamGelir"]-df["toplamGider"])
    df['sonuç'] = pd.to_numeric(df['sonuç'], errors='coerce')

    df["binarySonuc"]=0
    df['binarySonuc'] = pd.to_numeric(df['binarySonuc'], errors='coerce')
    df.loc[df['sonuç'] > 0, ['binarySonuc']] = 1
    df.binarySonuc
    return df
#%%Verileri dışarı aktarıyoruz
def editDataSetforSQLite():    
    df=createFeature() 
    df["durum"]="Zarar"
    df.loc[df['binarySonuc'] > 0, ['durum']] = "Kar"
    df.durum
    
    dumpFeatures = np.array(['Mazot','tohumGider', 'mazotGider', 'gübreGider', 'toplamGider', 'satışGelir','gübreVerimi', 'destekParasıGelir', 'toplamGelir','sonuç'])
    
    for i in range (len(dumpFeatures)):
        df = df.drop(dumpFeatures[i],axis=1)
    df.to_csv (r'C:\Users\Erdinc\Desktop\school\4\2.Summer\YMH414-BP(GP)\Codes\androidCodes\app\src\main\assets\product.csv', index = False, header=True)
    return df

#%%Korelasyon grafiği
def cGraphAgain():
    df = createFeature()
    listFeature = df.columns
    sns.heatmap(df[listFeature].corr(), annot = True , fmt = ".2f")
    plt.show()

#%%Dummies veriler ve One Hot Encoding Uygulaması
def makeDummies():
    df = createFeature()
    d =df.copy()
    #categoryVariableArr = ["Ürün","EkildiğiAy", "Bölge","ÜretimSüresi"]
    #for i in range (len(categoryVariableArr)):
    dDummies = pd.get_dummies(data=d, columns=["Ürün",'ÜretimSüresi',"EkildiğiAy", "Bölge"])
    dDummies
    #dMerged = pd.concat([d,dDummies],axis=1)
    #dMerged.columns = dMerged.columns.str.strip()
    return dDummies

def checkOut():
    df=makeDummies()
    dfDropFeature=['Üretim','Tohum Fiyatı', 'SatışFiyatı', 'Mazot', 'Gübre', 'EkilenAlan',
       'DestekParası', 'tohumGider', 'mazotGider','gübreGider', 'toplamGider', 'satışGelir', 'gübreVerimi',
       'destekParasıGelir', 'toplamGelir']
    for i in range(len(dfDropFeature)):
        df = df.drop(dfDropFeature[i],axis=1)
    df.columns
    return df

#%%Modelleme 
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import shuffle
def modelling():
    df=checkOut()
    label_encoder =LabelEncoder().fit(df.binarySonuc)
    labels = label_encoder.transform(df.binarySonuc)
    classes = list(label_encoder.classes_)
    
    
    y =df.drop("binarySonuc",axis=1)
    X=labels
    nb_features = 122
    nb_classes = len(classes)
    X, y = shuffle(X, y)
    
    #Variladtion 
    from sklearn.model_selection import train_test_split
    X_train,X_valid,y_train,y_valid=train_test_split(df, labels, test_size=0.3 )
    #Category Part
    from tensorflow.keras.utils import to_categorical
    y_train=to_categorical(y_train)
    y_valid=to_categorical(y_valid)
    nb_classes = len(classes)
    X_train = np.array(X_train).reshape(231, 122,1)
    X_valid = np.array(X_valid).reshape(99, 122,1)
    #Model
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Activation,SimpleRNN,Dropout,MaxPooling1D,Flatten,BatchNormalization,Conv1D
    import tensorflow as tf
    
    model = Sequential()
    model.add(Conv1D(512,1,input_shape=(nb_features,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256,1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2048,activation="relu"))
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(nb_classes,activation="sigmoid"))
    model.summary()

    #Compile and Training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m,recall_m])
    model.fit(X_train,y_train,epochs=50,validation_data=(X_valid,y_valid))
    print(model.evaluate(X_valid,y_valid,verbose=False))
    
    return model
"""
model = modelling()
import os

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(X_train,y_train,  
          epochs=10,
          validation_data=(X_train,y_train),
          callbacks=[cp_callback])  # Pass callback to training

model = modelling()
# Evaluate the model
loss, acc = model.evaluate(X_train,y_train, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss,acc = model.evaluate(X_train,y_train, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
"""
#%%
def staticsOfModelling():
    #Statics of Modelling
    model=modelling()
    print("Ortalama Eğitim Kaybı : ",np.mean(model.history.history["loss"]))
    print("Ortalama Eğitim Başarımı : ",np.mean(model.history.history["accuracy"]))
    print("Ortalama Doğrulama Kaybı : ",np.mean(model.history.history["val_loss"]))
    print("Ortalama Doğrulama Başarımı : ",np.mean(model.history.history["val_accuracy"]))
    print("Ortalama F1 - Skor Değeri : ",np.mean(model.history.history["val_f1_m"]))
    print("Ortalama Kesinlik Değeri : ",np.mean(model.history.history["val_precision_m"]))
    print("Ortalama Kesinlik Değeri : ",np.mean(model.history.history["recall_m"]))

    #Grafikler
    plt.plot(model.history.history["accuracy"])
    plt.plot(model.history.history["val_accuracy"])
    plt.title("Model Başarımları")
    plt.ylabel("Başarım")
    plt.xlabel("Epok")
    plt.legend(["Eğitim","Test"],loc="upper_left")
    plt.show()
"""
    plt.plot(model.history.history["loss"])
    plt.plot(model.history.history["val_loss"])
    plt.title("Model Kayıpları")
    plt.ylabel("Kayıp")
    plt.xlabel("Epok")
    plt.legend(["Eğitim","Test"],loc="upper_left")
    plt.show()
    
    plt.plot(model.history.history["f1_m"],color="g")
    plt.plot(model.history.history["val_f1_m"],color="r")
    plt.title("Model F1 Skorları")
    plt.ylabel("F1-Skorları")
    plt.xlabel("Epok Sayısı")
    plt.legend(["Eğitim","Doğrulama"],loc="uppper left")
    plt.show
    
    plt.plot(model.history.history["precision_m"],color="g")
    plt.plot(model.history.history["val_precision_m"],color="r")
    plt.title("Model Hasasiyeti Skorları")
    plt.ylabel("Hasasiyeti Skorları")
    plt.xlabel("Epok Sayısı")
    plt.legend(["Eğitim","Doğrulama"],loc="uppper left")
    plt.show
"""    
#editDataSetforSQLite()
#%%Modeli kaydetme 
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml


import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

model= modelling()
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
model.save('./model/my_model.h5')

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model('./model/my_model.h5')

# Let's check:
np.testing.assert_allclose(
  model.predict(X_train),
  reconstructed_model.predict(X_train))

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(X_train,y_train)

#%%
import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = ''
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)
"""   
#%%
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow.compat.v1 as tf
from past.builtins import xrange
import os
import os.path as path

def saveModel():
    TUTORIAL_NAME = 'out'
    MODEL_NAME = 'mnistTFonAndroid'
    SAVED_MODEL_PATH = './' + TUTORIAL_NAME+"/"
    
    
    
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)  
        saver = tf.train.Saver()
      
    saver.save(sess, SAVED_MODEL_PATH + MODEL_NAME + '.ckpt', global_step=i)  
    # Saving 
    tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
    tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb',as_text=False)        
      

    print("Saved Model")
    


#%%Main Fonksiyonu 
def main():
    #Veri Analizi
    print ("--Veri Analizi Bölümü-- \n\n\n")
    checkEmptyValues()
    summaryofDataset()
    
    #Veri Bilimi ve İstatistik Bölümü
    print ("--Veri Bilimi ve İstatistik Bölümü-- \n\n\n")  
    theoryofExampleforSeedPrice()
    descriptiveStatistics()    
    bernolliCase()
    believeCase()
    binomCase()
    possionCase()
    
    #Veri Önişleme
    print ("--Veri Önişleme Bölümü-- \n\n\n") 
    preProc()
    #preProc2()
    checkEmptyAreaAgain()       
    #cGraph()    
    
    #Veri Kümesi için Öznitelik Belirleme
    print ("--Veri Kümesi için Öznitelik Belirleme-- \n\n\n") 
    createFeature()
    editDataSetforSQLite()
    #cGraphAgain()
    makeDummies()
    
    #Veri Kümesi Modellenmesi
    print ("--Veri Kümesi Modellenmesi-- \n\n\n") 
    modelling()
    
    #Modelleme Sonuçları
    print ("--Model Sonuçları-- \n\n\n")
    staticsOfModelling()
    #saveModel()
    
#%%Main kısmını çalıştır
if __name__ == '__main__':
    #Veri Manipülasyonu Bölümü
    print ("--Veri Manipülasyonu Bölümü-- \n\n\n")
    df=readDatasetNinformantions()  
    
    main()



#%%Modelleme Karşılaştırmaları
"""
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3 ,random_state = 1) 

standard_scaler = StandardScaler()
Xtr_s = standard_scaler.fit_transform(X_train)
Xte_s = standard_scaler.transform(X_test)

robust_scaler = RobustScaler()
Xtr_r = robust_scaler.fit_transform(X_train)
Xte_r = robust_scaler.transform(X_test)
scaler = StandardScaler()

# Plot data
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].scatter(X_train[:, 0], X_train[:, 1],
              color=np.where(y_train > 0, 'r', 'b'))
ax[1].scatter(Xtr_s[:, 0], Xtr_s[:, 1], color=np.where(y_train > 0, 'r', 'b'))
ax[2].scatter(Xtr_r[:, 0], Xtr_r[:, 1], color=np.where(y_train > 0, 'r', 'b'))
ax[0].set_title("Unscaled data")
ax[1].set_title("After standard scaling (zoomed in)")
ax[2].set_title("After robust scaling (zoomed in)")
# for the scaled data, we zoom in to the data center (outlier can't be seen!)
for a in ax[1:]:
    a.set_xlim(-3, 3)
    a.set_ylim(-3, 3)
plt.tight_layout()
plt.show()


# Classify using k-NN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(Xtr_s, y_train)
acc_s = knn.score(Xte_s, y_test)
print("Testset accuracy using standard scaler: %.3f" % acc_s)
knn.fit(Xtr_r, y_train)
acc_r = knn.score(Xte_r, y_test)
print("Testset accuracy using robust scaler:   %.3f" % acc_r)
#%%
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 2)#.fit(X_train, y_train) 
knn.fit(X_test, y_test)
prediction = knn.predict(X_test)
print("{} nn Score : --> {}".format(2,knn.score(X_test, y_test))) #0.85

listOFScore = []
for i in range (1,50):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(X_test, y_test)
    knn2.score(X_test, y_test)
    listOFScore.append(knn2.score(X_test, y_test))

plt.plot(range(1,50),listOFScore)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

# KFold Cross Validation approach

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=knn,X=X_train,y=y_train,cv=2)

print("Ortalama Başarım",acc.mean()) #0.74
print("\n",acc.std()) #0.08

print(accuracy_score(y_test, knn.predict(X_test), normalize=True)*100)
 
# KFold Cross Validation approach
kf = KFold(n_splits=5,shuffle=False)
kf.split(X) 
accuracy_model = []
 
# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train the model
    model = knn.fit(X_train, y_train)
    # Append to accuracy_model the accuracy of the model
    accuracy_model.append(accuracy_score(y_test, knn.predict(X_test), normalize=True)*100)

     
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print(accuracy ) # 
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 

print("Ortalama Doğruluk : ",np.mean(accuracy_model))
grid = {"n_neighbors":np.arange(1,50)}

knnCv=GridSearchCV(knn,grid,cv=10)
knnCv.fit(X,y)
print("Tuned Hyperparametre",knnCv.best_params_)
print("Tuned Hyperparametre",knnCv.best_score_)
roc_auc_score(y_test, knn_predictions)
#%%
scores = pd.DataFrame(accuracy_model,columns=['Scores'])
 
sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="Scores",data=scores)
plt.show()
sns.set()
 
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state = 42) 
  
# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=gnb,X=X_train,y=y_train,cv=10)
print(acc.mean()) #0.89
print("\n",acc.std()) #0.07
gnb.fit(X_test, y_test)
print("--> ",gnb.score(X_test, y_test)) # 0.85
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print(accuracy ) # 0.92
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions) 

#%%
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
  
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=dtree_model,X=X_train,y=y_train,cv=10)

print(acc.mean()) 
print("\n",acc.std()) 
dtree_model.fit(X_test, y_test)
print("--> ",dtree_model.score(X_test, y_test))
cm = confusion_matrix(y_test, dtree_predictions) 

#%%Statics

print("Ortalama Eğitim Kaybı : ",np.mean(loss))
print("Ortalama Eğitim Başarımı : ",np.mean(accuracy))
print("Ortalama F1 : ",np.mean(f1_score))
print("Ortalana Kesinlik : ",np.mean(precision))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50)

print("Ortalama Eğitim Kaybı : ",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim Başarımı : ",np.mean(model.history.history["accuracy"]))
print("Ortalama Doğrulama Kaybı : ",np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama Kaybı : ",np.mean(model.history.history["val_accuracy"]))

test_results = model.evaluate(X_test,y_test)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')


plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Kazanım")
plt.xlabel("Epok")
plt.show
"""