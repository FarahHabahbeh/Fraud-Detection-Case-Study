#!/usr/bin/env python
# coding: utf-8

# In[50]:


pip install xgboost


# In[60]:


#importing packages
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


# In[52]:


#read file
dataset=pd.read_csv('C:\\Users\\PC\\My Case Study\\fraud.csv')


# In[53]:


#data cleansing
dataset["age"]=dataset["age"].str.strip("'")
dataset["gender"]=dataset["gender"].str.strip("'")
dataset["category"]=dataset["category"].str.strip("'")
dataset["category"] =dataset["category"].str[3:]


# In[54]:


#define type of variables
categorical_char_fields = ["age","gender","category"]
int_fields = ["fraud","step"]
float_fields = ["amount"]


# In[55]:


# assign proper data types
dataset.astype({field: 'int64' for field in int_fields}, copy=False)
dataset.astype({field: 'float64' for field in float_fields}, copy=False)
dataset.astype({field: 'category' for field in categorical_char_fields}, copy=False)


# In[56]:


#onehot encoder
categorical_data = dataset[categorical_char_fields]
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(categorical_data)
X_ohe = ohe.transform(categorical_data).toarray() # It returns a numpy array
##
###fix encoded column names###
fixed_column_names = []
for feature_name in ohe.get_feature_names_out():
    feature_index, feature_value = feature_name.split("_") 
    new_feature_name = feature_index+"_"+feature_value
    fixed_column_names.append(new_feature_name)
###end of fixing encoded column names###
##
ohe_data= pd.DataFrame(X_ohe,columns=fixed_column_names)
#print(ohe_data)
dataset_encoded = pd.concat([dataset.drop(categorical_char_fields, axis=1), ohe_data],axis=1)
#print(dataset)


# In[57]:


##training
training_data = dataset_encoded
X = np.array(training_data.drop(['customer','zipcodeOri','merchant','zipMerchant', 'fraud'], axis=1).values.tolist())
y = np.array(training_data['fraud'].values.tolist())
#define model
label=training_data["fraud"]
label_counts = label.value_counts()
pos_weight = label_counts[0] / label_counts[1]
model = XGBClassifier(max_depth=2, min_child_weight=1, scale_pos_weight=pos_weight, random_state=1234, nthread=10)
#fit the model
model.fit(X, y, eval_metric="auc")
#training accurcy
accuracy = model.score(X, y)
print("trainingaccuracy :",accuracy)


# In[61]:


##testholdout
#split the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
##
model.fit(X_train, Y_train, eval_metric="auc")
# Calculate and Print Training Accuracy
accuracy = model.score(X_train, Y_train)
print("Training Accuracy: %s" % accuracy)
# Predict on Testing Data
predictions = model.predict(X_test)
# Calculate and Print Confusion Matrix 
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
print("True Negatives: %s | False Positives: %s | False Negatives: %s | True Positives: %s" % (tn, fp, fn, tp))
# Calculate and Print Testing Accuracy
accuracy = accuracy_score(Y_test, predictions)
print("Testing Accuracy: %s" % accuracy)
# Calculate and Print Sensitivity
Sensitivity = recall_score(Y_test, predictions)
print("Sensitivity: %s" % Sensitivity)
# Calculate and Print precision
precision = precision_score(Y_test, predictions)
print("precision: %s" % precision)
# Calculate and Print Specificity
Specificity = tn/(tn+fp)
print("Specificity: %s" % Specificity)


# In[63]:


##preidict
##
Unencoded_target_data = dataset
Unencoded_target_data = Unencoded_target_data[Unencoded_target_data.fraud == 0]
Unencoded_target_data.reset_index(drop=True,inplace=True)
target_data = dataset_encoded[dataset_encoded.fraud == 0]
target_data.reset_index(drop=True,inplace=True)
#load data into array X
X = np.array(target_data.drop(['customer','zipcodeOri','merchant','zipMerchant', 'fraud'], axis=1).values.tolist())
##predict
model_predictions = model.predict_proba(X)
##add probability to unencoded file
Unencoded_target_data["prediction"] = model_predictions[:, 1]
#export file
Unencoded_target_data.to_csv(r'C:\\Users\\PC\\My Case Study\\Unencoded_target_data.csv')


# In[ ]:





# In[ ]:





# In[ ]:




