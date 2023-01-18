import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import joblib as jb

df = pd.read_csv('winequality-red.csv',delimiter=';')

x= df.iloc[:, :-1].values  
y= df.iloc[:, 11].values  

from sklearn.utils import resample

df_minority1 = df[df.quality==3] 
df_minority2 = df[df.quality==4] 
df_minority3 = df[df.quality==7] 
df_minority4 = df[df.quality==8] 
df_majority1 = df[df.quality==5] 
df_majority2= df[df.quality==6]


# Downsampling
df_majority_downsampled = resample(df_majority1, 
                                 replace=False,
                                 n_samples=400,
                                 random_state=123) 

df_majority_downsampled1 = resample(df_majority2, 
                                 replace=False,
                                 n_samples=400,
                                 random_state=123)
#Upsampling 
df_minority_up = resample(df_minority1, 
                        replace=True,
                        n_samples=400,
                        random_state=123) 

df_minority_up1 = resample(df_minority2, 
                        replace=True,
                        n_samples=400,
                        random_state=123) 

df_minority_up2 = resample(df_minority3, 
                        replace=True,
                        n_samples=400,
                        random_state=123) 
df_minority_up3 = resample(df_minority4, 
                        replace=True,
                        n_samples=400,
                        random_state=123) 

df_resampled = pd.concat([df_majority_downsampled,df_majority_downsampled1,df_minority_up,df_minority_up1, df_minority_up2,df_minority_up3])
df_resampled


x= df_resampled.iloc[:, :-1].values  
y= df_resampled.iloc[:, 11].values  

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0) 

sm = SMOTE(random_state=0)

clf = RandomForestClassifier(n_estimators=100,random_state=0)
model = RFE(clf,n_features_to_select=7)
x_sm , y_sm = sm.fit_resample(x_train,y_train)
x_test_sm , y_test_sm = sm.fit_resample(x_test,y_test)
model.fit(x_sm,y_sm)



filename = "redwine_classifier_model.joblib"
jb.dump(model, filename)
print("model saved successfully!")
