
# coding: utf-8

# # DIABETES PREDICTION

# ## Imports:

# In[154]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
#get_ipython().magic('matplotlib inline')
#sns.set()


# ## Load the Pima Indians Diabetes dataset:

# In[191]:


#load the pima indian diabetes dataset
diabetes=pd.read_csv("E:\Machine_Learning\DataSets\diabetes.csv")


# ## Inspect the dataset:

# In[196]:


'''print("diabetes shape is :",diabetes.shape)
print("Dataset Description:\n")
diabetes.describe()'''


# In[198]:


# Visualise a table with the first rows of the dataset, to better understand the data format
#print("Dataset head :\n")
#diabetes.head()


# ## Visualize the dataset:

# In[202]:


#diabetes.hist(bins=50, figsize=(20, 15))
#plt.show()


# In[95]:


#print the Outcome counts 1/0
#OutCount=diabetes.groupby("Outcome").size()
#print(OutCount)
#OutCount.plot(kind="bar",title="Outcome Count")


# ## Data correalation Matrix:

# In[93]:


#Finding Correlation of attributes with outcome
corr_mat=diabetes.corr()


# In[54]:


#correaltion matrix
corr_mat


# ### Visualize Predictors Correlation with Outcome:

# In[203]:


#plotting a graph for crrelation with Outcome
# here 8 is the index value of Outcome column
res=corr_mat.iloc[:-1,8]
#res.plot(kind='barh',title="Correlation Graph")


# #### From the above Correlation graph,It can be inferred that factors like "Age,BMI and BloodPressure" which can be measured without taking the blood sample ,ifluence the Outcome(0/1)

# # Data Cleaning and Transformation:

# #### There are some zero value records in the dataset.

# In[80]:

'''
zeros_Age=(diabetes["Age"]==0).sum()
zeros_BMI=(diabetes["BMI"]==0).sum()
zeros_BP=(diabetes["BloodPressure"]==0).sum()
print("Count of Zero values in Age : ",zeros_Age)
print("Count of Zero values in BMI : ",zeros_BMI)
print("Count of Zero values in BP : ",zeros_BP)'''


# #### Remove these records (zero value) from the dataset and create the required dataset for the model prediction.

# ## Creating Dataset for model:

# In[86]:


#temp_ds contains all non zero records of the diabetes dataset
temp_ds=pd.DataFrame(diabetes[(diabetes["Age"]>0) & (diabetes["BMI"]>0) & (diabetes["BloodPressure"]>0)])
main_dataset=pd.DataFrame(data=temp_ds,columns=["Age","BMI","BloodPressure","Outcome"])
'''print("Original dataset dimesnions(diabetes): ",diabetes.shape)
print("Original dataset without zero value records dimensions(temp_ds): ",temp_ds.shape)
print("Dataset for Model without zero value records dimensions(main_dataset): ",main_dataset.shape)'''


# In[105]:


#main_dataset.describe()


# #### main_dataset contains 729 non zero records.

# In[111]:


#out_count=main_dataset.groupby("Outcome").size()
#print(out_count)
#out_count.plot(kind="bar",title="Outcome label Count in main Dataset")


# # Splitting the Dataset:

# In[115]:


#feature matrix
X=main_dataset.iloc[:,:-1]
#X.head()
#X_train,X_test,y_train,y_test=train_test_split(main_dataset,random_state=66)


# In[116]:


#value vector
y=main_dataset["Outcome"]
#y.head()


# In[123]:


# Split the training dataset in 80% / 20%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=60,stratify=y)
'''print("X_train shape : ",X_train.shape)
print("y_train shape : ",y_train.shape)
print("X_test shape : ",X_test.shape)
print("y_test shape : ",y_test.shape)'''


# ## Feature Scaling:

# In[147]:


scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)
#print("type(X_train_scaled) : ",type(X_train_scaled))
#print("type(X_test_scaled) : ",type(X_test_scaled))


# ### Scaled Values:

# In[148]:


#create a pandas dataframe to display the scaled values
#sv=pd.DataFrame(data=X_train_scaled)
#sv.head()


# # Training the SVM model:

# In[150]:


#create instance for SVC
svc=SVC()
svc.fit(X_train_scaled,y_train)


# ### Check Accuracy:

# In[158]:


# use score of SVC() to find Accuracy
#train_accuracy=svc.score(X_train_scaled,y_train)
#test_accuracy=svc.score(X_test_scaled,y_test)
#print("Accuracy on training set: ",train_accuracy)
#print("Accuracy on testing set: ",test_accuracy)


# ## Model Tuning:

# ### Find the best Parameters for SVC.

# In[157]:


param_grid = {
    'C': [1.0, 10.0, 50.0],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'shrinking': [True, False],
    'gamma': ['auto', 1, 0.1],
    'coef0': [0.0, 0.1, 0.5]
}

model_svc = SVC()

grid_search = GridSearchCV(model_svc, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)


# #### Print the best score found by GridSearchCV:

# In[162]:


best_score=grid_search.best_score_
#print("Best score = ",best_score)


# #### Apply the Parameters to the Model and train it:

# In[165]:


# Create an instance of the SVC algorithm using parameters
# from best_estimator_ property
best_svc = grid_search.best_estimator_
#train the model
best_svc.fit(X_train_scaled,y_train)


# ### Check Accuracy:

# In[167]:


# use score of SVC() to find Accuracy
'''best_train_accuracy=best_svc.score(X_train_scaled,y_train)
best_test_accuracy=best_svc.score(X_test_scaled,y_test)
print("Best Accuracy on training set: ",best_train_accuracy)
print("Best Accuracy on testing set: ",best_test_accuracy)'''


# ## Make a Prediction:

# In[169]:


# create a new (fake) person by taking the values of Age,BMI and BloodPressure
'''new_person = pd.DataFrame([[50,33.6,72]])
# Scale those values like the others using MinMaxScaler
new_person_scaled = scaler.transform(new_person)


# In[177]:


#predict the outcome
#here "1" means "person is likely to have type-2 diabetes"
# 0 means "person doesn't have type-2 diabetes
prediction = best_svc.predict(new_person_scaled)
type(prediction)


# In[176]:


print("Prediction value : ",prediction[0])


# In[178]:


if(prediction==1):
    print("You are likely to have type-2 diabetes.")
else:
    print("Congratulations, You don't have type-2 diabetes.")
'''

# ### Taking the input from user:

# In[179]:

def predict(age,bmi,bp):
    '''age = (int)(input("Age : "))
    BMI = (float)(input("BMI : "))
    BP = (int)(input("Blood Pressure : "))'''
    #pred = input(("Enter Y to predict : "))


    # In[187]:


    person = pd.DataFrame([[age,bmi,bp]])
    person_scaled = scaler.transform(person)


# In[190]:


#if(pred == "Y" or pred == "y"):
    P_prediction = best_svc.predict(person_scaled)

    return P_prediction;
    '''if(P_prediction==1):
        print("You are likely to have type-2 diabetes.")
    else:
        print("Congratulations, You don't have type-2 diabetes.")
else:
    print("You did not entered Y or y ")'''

