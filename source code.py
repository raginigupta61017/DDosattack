#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#
dataset =pd.read_csv('/content/sample_data/IDS_dataset.csv')
X=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

def Multilabelencoder(X,k):
    from sklearn.preprocessing import LabelEncoder
    X[:,k]= LabelEncoder().fit_transform(X[:,k])
    return X

for i in range(1,4):
    X=Multilabelencoder(X,i)
#
from sklearn.preprocessing import LabelEncoder
y= LabelEncoder().fit_transform(y)
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)
#
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#
import time
s=time.time()

from sklearn.svm import SVC
classifier1 = SVC(kernel ='linear', random_state=0)
classifier1.fit(X_train,y_train)

from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators=20, criterion='entropy',random_state=0)
classifier3.fit(X_train,y_train)

e=[classifier1,classifier2,classifier3]

X_first=X_test
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance= pca.explained_variance_ratio_

from sklearn.svm import SVC
classifier1 = SVC(kernel ='linear', random_state=0)
classifier1.fit(X_train,y_train)

from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators=20, criterion='entropy',random_state=0)
classifier3.fit(X_train,y_train)
    
f=[classifier1,classifier2,classifier3]

g=["Support Vector Machine - linear","Gaussian Naive Bayes","random forest"]

def abc(h):
    warnings.filterwarnings("ignore")
    classifier=e[h]
    Y_pred = classifier.predict(X_first)
    #
    from  sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,Y_pred)
    #
    print("Accuracy of the "+g[h]+" Model is : ",(cm[0][0]+cm[1][1])*100/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]))
    print("Precision of the "+g[h]+" Model is : ",(cm[0][0])*100/(cm[0][0]+cm[1][0]))
    print("Recall of the "+g[h]+" Model is : ",(cm[0][0])*100/(cm[0][0]+cm[0][1]))
    #
    classifier=f[h]
    #
    Y_pred = classifier.predict(X_test)
    #
    from matplotlib.colors import ListedColormap
    X_set,y_set=X_test,y_test
    X1,X2= np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
    plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap =ListedColormap(('red','green')))
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c = ['red', 'green'][i], label = j)
    plt.title('PDC for Prediction using '+g[h])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
    
import multiprocessing as mp
def master():
    if __name__=='__main__':
        while(1):
            warnings.filterwarnings("ignore")
            s=int(input("""\nWelcome to DDOS attack Network Detection System
                    Make your choice :
                    0. Support Vector Machine - Linear
                    1. Naive Bayes
                    2. Random Forest
                    3. Run all Algorithms
                    4. Quit\n"""))
            if(s<4):
                abc(s)
            elif(s==4):
                s=int(input("""Select how you want to run it :
                                0. Serially
                                1. Parallelly
                                2. Go Back\n"""))
                if(s==0):
                    for i in range(4):
                        abc(i)
                elif(s==1):
                    parallel()
                elif(s==2):
                    pass;
                else:
                    print("INVALID CHOICE :( TRY AGAIN")
            elif(s==5):
                break;
            else:
                print("INVALID CHOICE :( TRY AGAIN")
    else:
        pass

def parallel():
    p = mp.Pool(4)
    for h in range(4):
        p.apply_async(abc, args=(h,))
    p.close()
    p.join()

master()