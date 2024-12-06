
# customer churn analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('Churn_Modelling.csv')

#veri on isleme
X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values


#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#3 Yapay Sinir ağı
from tensorflow.keras.models import Sequential # yapay sinir ağı
from tensorflow.keras.layers import Dense # katmanları


classifier = Sequential()

# İlk gizli katman
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
# input_dim: Giriş katmanındaki bağımsız değişken sayısı (11)
# units: Katmandaki nöron sayısı (6) (genellikle giriş değişkenlerinin ortalaması)
# kernel_initializer: 'uniform' ile ağırlıklar -0.05 ile 0.05 arasında rastgele atanır
# activation: Aktivasyon fonksiyonu olarak ReLU (Rectified Linear Unit) kullanılır, negatif değerler 0'a sabitlenir

# İkinci gizli katman
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
# Gizli katmanda tekrar aynı sayıda nöron (6) kullanılır, ağırlıklar aynı yöntemle başlatılır
# ReLU aktivasyonu ile negatif girdiler sıfırlanır, pozitif değerler aynen aktarılır

# Çıkış katmanı
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Çıkış katmanında 1 nöron vardır çünkü ikili sınıflandırma yapılmaktadır (örneğin: 0 veya 1)
# Aktivasyon fonksiyonu olarak sigmoid kullanılır, çıktı değeri 0 ile 1 arasında bir olasılık verir

# Modelin derlenmesi
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer: 'adam', adaptif öğrenme oranına sahip gelişmiş bir gradyan inişi algoritmasıdır
# loss: İkili sınıflandırma için uygun olan 'binary_crossentropy' kayıp fonksiyonu kullanılır
# metrics: Eğitim sırasında modelin başarısını ölçmek için doğruluk (accuracy) metriği kullanılır


classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_pred,y_test)
print(cm)
print(ac)

