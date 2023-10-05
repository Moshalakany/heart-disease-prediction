#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Moshalakany/heart-disease-prediction/blob/main/heart_disease_predection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[106]:
# colab link-->"https://colab.research.google.com/github/Moshalakany/heart-disease-prediction/blob/main/heart_disease_predection.ipynb"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.metrics import confusion_matrix, mean_squared_error,accuracy_score,classification_report,precision_recall_curve,PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


# In[107]:


data=pd.read_csv(r"/content/drive/MyDrive/data/Heart_Disease.csv")


# In[108]:


data.describe()


# In[109]:


print(data)


# In[109]:





# In[110]:


data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Gender'].fillna(data['Gender'].mode(),inplace=True)
data['work_type'].fillna(data['work_type'].mode(),inplace=True)
data['smoking_status'].fillna(data['smoking_status'].mode(),inplace=True)


# In[111]:


#to replac unknown values in smoking_status column with the mod value
data.replace(to_replace="Unknown",value='never smoked',inplace=True)
  


# In[112]:


print(data['smoking_status'].mode())


# In[113]:


dummy_cols=['work_type','smoking_status','Gender' ]
non_dummy_cols=list(set(data.columns)-set(dummy_cols))


# In[114]:


data=pd.get_dummies(data,columns=dummy_cols)
data['Heart Disease']=data['Heart Disease'].replace("Yes",1)
data['Heart Disease']=data['Heart Disease'].replace("No",0)
     
print(data)


# In[115]:


print(data.isnull().sum()) 


# In[116]:


data.duplicated() 


# In[117]:


data[(data["Cholesterol"]>370)] 


# In[118]:


sns.boxplot(data=data,palette='rainbow',orient='h')
# cholestrol oultiers remove
data.drop([  1,   9,  181,52,188],axis=0,inplace=True)



# In[119]:


print('skewness value of Cholestrol:',data['Cholesterol'].skew())
# 1.1628003276247532 before remove outlier
 
   


# In[120]:


from scipy import stats
z=np.abs(stats.zscore(data['Cholesterol']))
print(z)
#  zscore 


# In[121]:


corr_matrix = data.corr()
print(abs(corr_matrix["Heart Disease"]).sort_values(ascending=False))


# In[122]:


sns.heatmap(data.corr())
data.drop('work_type_Private',axis=1,inplace=True)
data.drop('id',axis=1,inplace=True)
data.drop('BP',axis=1,inplace=True)
data.drop('FBS over 120',axis=1,inplace=True)
data.drop('work_type_Never_worked',axis=1,inplace=True)
data.drop('work_type_Self-employed',axis=1,inplace=True)
data.drop('work_type_children',axis=1,inplace=True)
data.drop('smoking_status_smokes',axis=1,inplace=True)
data.drop('smoking_status_never smoked',axis=1,inplace=True)
data.drop('smoking_status_formerly smoked',axis=1,inplace=True)


# In[123]:


data.corr()


# In[124]:


print(data.duplicated().sum())


# In[125]:


x=data.drop('Heart Disease',axis=1) 
y=data['Heart Disease']


# In[126]:


from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
x=scaler.fit_transform(x)
x=scaler.transform(x)
print (x)


# In[127]:


data.head() 


# In[128]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)


# In[129]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',C=1,random_state = 3)
# hyperparameter=random_state
classifier.fit(X_train, y_train)


# In[130]:


predict=classifier.predict(X_test)
print(predict)


# In[131]:


mean_squared_error(y_test,predict)
cm = confusion_matrix(y_test, predict, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()


# In[132]:


report=classification_report(y_test,predict)
print(report)


# In[133]:


classifier.score(X_train,y_train)


# In[134]:


accuracy_score(y_test,predict)


# In[135]:


report=classification_report(y_test,predict)
print(report)


# In[136]:


precision, recall, _ = precision_recall_curve(y_test, predict)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()


# In[137]:


for col in data.columns:
 fig=plt.figure(figsize=(2,2))
 plt.title(col)
 plt.hist(data[col])


# In[138]:


from sklearn.model_selection import train_test_split
Xt, Xte, yt, y_pred = train_test_split(x, y, test_size = 0.22, random_state = 3)


# In[139]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', C=3,random_state = 4)
classifier.fit(Xt, yt)


# In[140]:


pred=classifier.predict(Xte)
print("prediction", pred)


# In[141]:


cm = confusion_matrix(y_pred, pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
mean_squared_error(y_pred, pred)


# In[142]:


report=classification_report(y_pred,pred)
print(report)


# In[143]:


precision, recall, _ = precision_recall_curve(y_pred, pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()


# In[144]:


accuracy_score(y_pred,pred)


# In[145]:


classifier.score(Xt,yt)


# In[146]:


report=classification_report(y_pred,pred)
print(report)


# In[147]:


sns.pairplot(data,vars=['Max HR', 'Age' ],diag_kind='scatter',hue='Heart Disease',markers=["o","s"])
plt.show()


# In[148]:


from sklearn.model_selection import train_test_split
Xdst_train, Xdst_test, ydst_train, ydst_test = train_test_split(x, y, test_size = 0.2, random_state=6)


# In[149]:


from sklearn.tree import DecisionTreeClassifier
classifir = DecisionTreeClassifier(criterion = 'entropy',random_state = 4,max_depth=4,max_features=3)
classifir.fit(Xdst_train, ydst_train)


# In[150]:


ydst_pred = classifir.predict(Xdst_test)
print(ydst_pred)


# In[151]:


cm = confusion_matrix(ydst_test, ydst_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()

mean_squared_error(ydst_test, ydst_pred)


# In[152]:


accuracy_score(ydst_test,ydst_pred)


# In[153]:


report=classification_report(ydst_test,ydst_pred)
print(report)


# In[154]:


precision, recall, _ = precision_recall_curve(ydst_test, ydst_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()


# In[155]:


classifir.score(Xdst_train,ydst_train)


# In[156]:


from sklearn.model_selection import train_test_split
Xknn_train, Xknn_test, yknn_train, yknn_test = train_test_split(x, y, test_size = 0.2, random_state =3)


# In[157]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =4 , metric = 'minkowski', p = 1)
# Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1),
classifier.fit(Xknn_train, yknn_train)


# In[158]:


yknn_pred = classifier.predict(Xknn_test)
print(yknn_pred)


# In[159]:


cm = confusion_matrix(yknn_test, yknn_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
accuracy_score(yknn_test, yknn_pred)


# In[160]:


mean_squared_error(yknn_test, yknn_pred)


# In[161]:


classifir.score(Xknn_train,yknn_train)
# train acurracy


# In[162]:


report=classification_report(yknn_test,yknn_pred)
print(report)


# In[163]:


precision, recall, _ = precision_recall_curve(yknn_test, yknn_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()


# In[164]:


from sklearn.model_selection import train_test_split
Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(x, y, test_size = 0.2, random_state =6 )


# In[165]:


classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state =4 ,max_depth=4,max_features='sqrt')             
classifier.fit(Xrf_train, yrf_train)


# In[166]:


yrf_pred = classifier.predict(Xrf_test)
print(yrf_pred)


# In[167]:


cm = confusion_matrix(yrf_test, yrf_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
accuracy_score(yrf_test, yrf_pred)


# In[168]:


mean_squared_error(yrf_test, yrf_pred)


# In[169]:


classifir.score(Xrf_train,yrf_train)


# In[170]:


report=classification_report(yrf_test,yrf_pred)
print(report)


# In[171]:


precision, recall, _ = precision_recall_curve(yrf_test, yrf_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()


# In[182]:


print(data)


# In[173]:


data.head()


# In[174]:


get_ipython().system(' pip3 install xgboost')


# In[175]:


from sklearn.model_selection import train_test_split
Xgb_train, Xgb_test, ygb_train, ygb_test = train_test_split(x, y, test_size = 0.3, random_state =4 )


# In[176]:


import xgboost as xgb
classifier = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='auto', eta=0.03, max_depth=3)
classifier.fit(Xgb_train, ygb_train)


# In[177]:


ygb_pred = classifier.predict(Xgb_test)
print(ygb_pred)


# In[178]:


cm = confusion_matrix(ygb_test, ygb_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
print("testing accuracy =",accuracy_score(ygb_test, ygb_pred))
print("training accuracy =",classifir.score(Xgb_train,ygb_train))


# In[179]:


report=classification_report(ygb_test,ygb_pred)
print(report)


# In[180]:


precision, recall, _ = precision_recall_curve(ygb_test, ygb_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()

