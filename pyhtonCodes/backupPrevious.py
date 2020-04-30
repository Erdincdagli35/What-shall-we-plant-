# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 01:27:56 2020

@author: Erdinc
"""

    
#%%    
    
    #We can use it if it exist any empty value
    #df[""].fillna(0,inplace = True)
    #Summary of dataset 
"""
katDf = df.select_dtypes(include= ["object"])
katDf.head(5)
katDf
#Selection 
dfNum = df.select_dtypes(include=["float64","int64","object"])
dfNum.head()
"""
#%%Category Case 
df["Üretim"].value_counts()

from pandas.api.types import CategoricalDtype
#df.PlantedMounth = df.PlantedMounth.astype(CategoricalDtype(ordered = True))
# for plantedMountCategory 

productionTimeCategory = ["Kısa Süre","Uzun Süre"]
df.ProductionTime = df.ProductionTime.astype(CategoricalDtype(categories =productionTimeCategory ,ordered = True))

#%%Barplot
(df["Region"]
            .value_counts()
            .plot.barh()
            .set_title("Bölgelere Tarım Ürünlerin Dağılımlarını"));

(df["PlantedMounth"]
            .value_counts()
            .plot.barh()
            .set_title("Ekim Aylarına Göre Tarım Ürünlerin Dağılımlarını"));
                      
sns.barplot(x="Region" ,y=df.Region.index , data = df)

#%%Cross

#%%Histogram 
sns.distplot(df.Production , bins =10, kde=False)
#?sns.distplot

#%%Histogram and Crossing
     
#sns.catplot(x = "SeedPrice",y="Region",hue="color" , kind="point",data=df)

#%%Boxplot
sns.boxplot(x=df["SeedPrice"]);#boxplot as horizontal 
sns.boxplot(x=df["SeedPrice"] , orient = "v"); #boxplot as Vertical 

#%%Boxplot Crossing
#In which month was the best production done?  
sns.boxplot(x="PlantedMounth",y="Production",data=df);
#In which plantedTime was the best production done ? 
sns.boxplot(x="ProductionTime",y="Production",data=df);

#%%Violin Graph
sns.catplot(y="Production",kind="violin",data=df)

#%%Violin Crossing
sns.catplot(x="Region",y="Production",hue="SeedPrice",kind="violin",data=df)

#%%Scatterplot
sns.scatterplot(x="Production" , y= "Region",data=df);

#%%Scatterplot crossing
sns.scatterplot(x="Production" , y= "Region", hue="PlantedMounth" ,data=df);
sns.scatterplot(x="Production" , y= "Region", hue="PlantedMounth",style="Region" ,data=df);

#%%with lmplot
sns.lmplot(x="Production" , y= "SeedPrice",hue="Region",data=df)
sns.lmplot(x="Production" , y= "SeedPrice",hue="Region",col="ProductionTime",data=df)

#%%Scatterplot Matrix
sns.pairplot(df)
sns.pairplot(df , hue = "ProductionTime")

#%%Line Graphing
sns.lineplot(x="Production" , y= "SeedPrice",data=df);
sns.lineplot(x="Production" , y= "SeedPrice",hue="Region",style="Region",markers=True,dashes=False,data=df);

#%%# DATA SCIENCE AND STATISTICS - Theory of Example


#%%


#%%

