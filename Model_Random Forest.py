import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#loading data
df = pd.read_csv('C:\\Users\\rajat\\Desktop\\PredictFailure.csv',names=['failure', 'attribute1', 'attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute8','attribute9'])
# removing the first row of data as it was name of column
df=df[1:]
# there were two 0 and two 1, so grouped them together
df['failure']=[1 if b==1 or b=='1' else 0 for b in df.failure]
#divided data into target and features
y=df.failure
x=df.drop('failure',axis=1)
#split the data into training and testing sets
x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(x,y,test_size=0.25)
#For standardizing data


print("***************************Prediction on Original Data****************************\n")
print("             @@@@@@@@@@@Original Accuracy and Results@@@@@@@@@@@\n")
clf = RandomForestClassifier(max_depth=15, random_state=1)
clf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)
print("Accuracy =", accuracy_score(y_test_original,predictions))
print(np.unique(predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
print("true negative",tn)
print("false positive",fp)
print("false negative",fn)
print("true positive",tp)
#print("Accuracy= ",accuracy_score(predictions,y_test_original))
#print("Classes printed by model= ",np.unique(predictions))
#print("Classification Report=\n",classification_report(y_test_original,predictions))

print("***************************UPSAMPLING THE MINORITY CLASS****************************\n")
#dividing data into two sets, minority(1) and majority(0)
df_majority=df[df.failure==0]
df_minority=df[df.failure==1]
#increasing the size of minroty and making it equal to majority
df_minority_upsampled=resample(df_minority,n_samples=123488, random_state=123)
#combining the upsampled minority and original majority
df_upsampled=pd.concat([df_majority,df_minority_upsampled])
upsampled_y=df_upsampled.failure
upsampled_x=df_upsampled.drop('failure',axis=1)
x_train_upsample,x_test_upsample,y_train_upsample,y_test_upsample=train_test_split(upsampled_x,upsampled_y,test_size=0.25)
clf = RandomForestClassifier(max_depth=15, random_state=1)
print("            @@@@@@@@@@@Accuracy and Results on Upsampled data@@@@@@@@@@@\n")
clf.fit(x_train_upsample,y_train_upsample)
predictions=clf.predict(x_test_upsample)
print("Accuracy= ",accuracy_score(predictions,y_test_upsample))
print("Classes printed by model= ",np.unique(predictions))
print("Classification Report=\n",classification_report(y_test_upsample,predictions))

#testing the model generated on upsampled data on the original imbalanced data
print("        @@@@@@@@@@@Accuracy Original data trained in upsample model@@@@@@@@@@@\n")
predictions=clf.predict(x_test_original)
print("Accuracy= ",accuracy_score(predictions,y_test_original))
print("Classes printed by model= ",np.unique(predictions))
print("Classification Report=\n",classification_report(y_test_original,predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
print("true negative",tn)
print("false positive",fp)
print("false negative",fn)
print("true positive",tp)

print("***************************DOWNSAMPLING THE MINORITY CLASS****************************\n")
#reducing the majority class to size of minority class
df_majority_downsampled=resample(df_majority,n_samples=106,random_state=123)
df_downsampled=pd.concat([df_majority_downsampled,df_minority])
downsampled_y=df_downsampled.failure
downsampled_x=df_downsampled.drop('failure',axis=1)
x_train_downsample,x_test_downsample,y_train_downsample,y_test_downsample=train_test_split(downsampled_x,downsampled_y,test_size=0.25)
#clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,min_samples_leaf=5)
clf = RandomForestClassifier(max_depth=15, random_state=1)
clf.fit(x_train_downsample,y_train_downsample)
predictions=clf.predict(x_test_original)
print("        @@@@@@@@@@@Accuracy Original data trained in downsample model@@@@@@@@@@@\n")
print("Accuracy= ",accuracy_score(predictions,y_test_original))
print("Classes printed by model= ",np.unique(predictions))
print("Classification Report=\n",classification_report(y_test_original,predictions))
tn, fp, fn, tp = confusion_matrix(y_test_original,predictions).ravel()
print("true negative",tn)
print("false positive",fp)
print("false negative",fn)
print("true positive",tp)