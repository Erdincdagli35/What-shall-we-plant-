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
    return df

import random
from random import seed

def setAreaColumnType():
    df=readDatasetNinformantions()
    df['EkilenAlan'] = pd.to_numeric(df['EkilenAlan'], errors='coerce')
    return df

#setAreaColumnType()
#%%Verilerin boş değerlerini tespit etmemiz gereken alan -- Veri Analizi Bölümü
def checkEmptyValues():
    df = setAreaColumnType()
    #Is any value empty ? 
    if df.isnull().values.any() == True:
         df=df.dropna()
         return df
    else:
        return df

#%%datasetimizin özeti belirleyebilmek için bir fonksiyon yazacağız
def summaryofDataset():
    df=checkEmptyValues()
    print("Üretim : Avg : "+str(df["Üretim"].mean()))
    print("Tohum Fiyatı : Avg : "+str(df["Tohum Fiyatı"].mean()))
    print("SatışFiyatı : Avg : "+str(df["SatışFiyatı"].mean()))
    print("Mazot : Avg : "+str(df["Mazot"].mean()))
    print("Gübre : Avg : "+str(df["Gübre"].mean()))
    print("Aylara göre ekim yapılma dağılımları : "+str(df["EkildiğiAy"].value_counts())) 

#%%datasetimizi  grafiklerimizi oluşuturuyoruz
def crossingWithScatterplot():
    df=checkEmptyValues()  
    sns.scatterplot(x="Üretim" , y= "Bölge", hue="Tohum Fiyatı",size="Tohum Fiyatı" ,data=df);

crossingWithScatterplot()

#%%Örnek Teorisi - Veri Bilimi ve İstatislik Bölümü 
#Rastgele seçilen bir sütunda nasıl bir ortalama çıkaracağı sonucunu öğrenmek için 
#Örneklem teorisine başvurduk.
def theoryofExampleforSeedPrice():
    df=checkEmptyValues()
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
theoryofExampleforSeedPrice()

#%%Betimsel İstatistik Uygulamaları
import researchpy as rp
def descriptiveStatistics():
    df=checkEmptyValues()
    rp.summary_cont(df[["Üretim","Tohum Fiyatı"]])
    rp.summary_cat(df[["EkildiğiAy","Bölge","ÜretimSüresi"]])
    df[["Üretim","EkilenAlan"]].cov()

descriptiveStatistics()    

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
preProc2()
    
#%%empty area for category veriable 
def checkEmptyAreaAgain():
    df=checkEmptyValues()
    df.isnull()
    df.groupby("EkildiğiAy")["EkildiğiAy"].count()
    df.groupby("Bölge")["Bölge"].count()
    df.groupby("ÜretimSüresi")["ÜretimSüresi"].count()

#%%Koalasyon grafiği
def cGraph():
    df=checkEmptyValues()
    listFeature = ["Ürün", "Üretim", "Tohum Fiyatı",'SatışFiyatı', 'Mazot', 'Gübre', 'EkilenAlan']
    sns.heatmap(df[listFeature].corr(), annot = False , fmt = ".2f")
    plt.show()
cGraph()
#%%Giridilerin alınması 
def getInputs():
    m2 = int(input("Ekilecek Alanını giriniz (Hektar): "))
    region = str(input("Ekilecek Alanını Bölgesini Giriniz (Ege - Akdeniz - Karadeniz - GDoğu Anadolu - Marmara - Doğu Anadolu) : "))
    pTime= str(input("Ekilecek Alanın üretim süresini giriniz : (Kısa Süre - Uzun Süre) : "))
    mounth=str(input("Ekilecek Alanın Hangi ayda eklieceğini girin : (Ocak - Şubat - .. - Aralık) : "))
    felitizer=bool(input("Ekilecek Alanın Gübre atılacak mı  : (true - false) :"))
    return m2,region,pTime,mounth,felitizer

#%%Kar zarar durum Feature umuz için değişkenllerimizin son düzenlemelerini yağacağız
#Index(['Ürün', 'Üretim', 'EkildiğiAy', 'Tohum Fiyatı', 'Bölge', 'ÜretimSüresi','SatışFiyatı', 'Mazot', 'Gübre', 'EkilenAlan']       
df=checkEmptyValues()
M2,Region,PTime,Mounth,Felitizer=getInputs()

#Giderler
#Tohum Fiyatı,Mazot,Gübre

#Üretim Oranı
#m2 başına ton bazında üretim
df["Oran"] = (df["EkilenAlan"]/10000) / df["Üretim"]

#Tohum Fiyatı
#Genellikle hektar başına ortalama 100 – 160 kg tohum gerekir
randomSeedNumber = random.randint(100, 160)
df["tohumGider"]=M2*df["Tohum Fiyatı"]*randomSeedNumber

#Mazot Fiyatı
#Traktörümüzü 1 Saatlik bir çalışmada 0.203 dan 0.24 lt mazot yakıyor
randomDieselNumber = random.uniform(0.203, 0.24)
df["mazotGider"]= df["Mazot"]*3*5*3#ortalama 3-4 aylık üretime sahip ürünler içinrandomDieselNumber#Ortalama 3 ay, ortalama 15 gün , ortalama 3 saat
df["mazotGider"][df.ÜretimSüresi=="Uzun Süre "]=df["Mazot"][df.ÜretimSüresi=="Uzun Süre "]*6*10*5*randomDieselNumber# ortalama 5-8 aylık üretime sahip ürünler için : Ortalama 6 ay, ortalama 20 gün , ortalama 5 saat

#Gübre Fiyatı
#Uzun Süreli ürünlere 70-80kg ,kısa süreli ürünlere 20-40kg gübre atılır
#1 Gübre torbasına(50kg) göre fiyatlandırılmış sütunumuzu seçicez 
randomFelitizerNumberforLongT = random.randint(70, 80)
randomFelitizerNumberforShortT = random.randint(20, 40)
df["gübreGider"]= df["Gübre"]*randomFelitizerNumberforShortT
df["gübreGider"][df.ÜretimSüresi=="Uzun Süre "]=df["Gübre"][df.ÜretimSüresi=="Uzun Süre "]*randomFelitizerNumberforLongT

#↑Toplam Gideri Hesaplayalım
df["toplamGider"]=df["tohumGider"]+df["mazotGider"]+df["gübreGider"]

#Gelirler
#Satış Fiyatı
df["satışGelir"]=df["SatışFiyatı"]*(M2/df["Oran"])

#Çiftçi Destek Parası
df["destekParasıGelir"]=df["DestekParası"]*M2

#Toplam Gelir
df["toplamGelir"]= df["destekParasıGelir"]+df["satışGelir"]

#Kar - Zarar Durumu 
df["sonuç"]=df["toplamGelir"]-df["toplamGider"]

#%%Kar zarar Durum Feature mızı oluşutracağız
"""

"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

df = df.sample(frac=1).reset_index(drop=True)
X=df.iloc[:,1:]
y=df.iloc[:,0]
le = preprocessing.LabelEncoder()
le.fit(y)
y =le.transform(y)

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train,y_train)
print(lr_clf.predict(X_test))
print(lr_clf.score(X_test,y_test))

#%%Main 
"""
    getInputs(m2,region,pTime,mounth,felitizer)
readDatasetNinformantions()
setAreaColumnType()
"""


