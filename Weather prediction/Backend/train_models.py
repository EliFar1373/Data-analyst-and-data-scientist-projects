
import numpy as np
import pandas as pd
import matplotlib.pyplot as splt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[4]:


data=pd.read_csv(r"C:\Users\elham\OneDrive\Desktop\IT project\Data analyst and scientist projects\Weather prediction\weather_data.csv")


# In[5]:


data.head(2)


# In[6]:


data.info()


# In[7]:


len(data)


# In[9]:


data=data.dropna()


# In[10]:


len(data)


# In[11]:


data.describe()


# In[12]:


data.columns


# # String Indexing

# In[ ]:





# In[14]:


data['Location_index']=data['Location'].astype('category').cat.codes
data.head(2)


# In[18]:


# Create a new DataFrame with both columns
unique_location=data['Location'].drop_duplicates().reset_index(drop=True)
unique_Location_index=data['Location_index'].drop_duplicates().reset_index(drop=True)
unique_df=pd.DataFrame({"unique_location":unique_location,"unique_Location_index":unique_Location_index})
unique_df


# In[49]:


list(unique_location)


# In[23]:


X=data.drop(['Location','Temperature (°C)'],axis=1)
Y=data['Temperature (°C)']
X.columns


# In[24]:


len(X)


# In[41]:


# Select numeric columns
col_numeric=data.select_dtypes(include='number').columns
col_numeric

col_numeric_corr=data[col_numeric].corr()
import matplotlib.pyplot as plt
import seaborn as sns 
plt.figure(figsize=(8,8))
sns.heatmap(col_numeric_corr,annot=True,fmt='.2f',cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# # Feature selection

# In[47]:


from sklearn.feature_selection import SelectKBest,f_regression
# Select top 5 features
selector=SelectKBest(score_func=f_regression,k=5)
New_x=selector.fit_transform(X,Y)
# Get the selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected features:", list(selected_features))


# In[57]:


X_selected_feature=data[selected_features]
X_selected_feature.head(2)


# In[58]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_selected_feature,Y)


# # Scaling:

# In[69]:


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)


# In[70]:


df_scaled = pd.DataFrame(X_train_scaled, columns=X_selected_feature.columns)
print(df_scaled.describe())


# In[71]:


df_scaled.head(2)


# # Train Linear Regression Model['Regression', 'Lasso', 'Ridge']

# In[80]:


#'Regression', 'Lasso', 'Ridge'
from sklearn.linear_model import Lasso,Ridge


score={}
# Define models
models = {
    "model_reg": LinearRegression(),
    "model_ridge": Ridge(alpha=1.0),
    "model_lasso": Lasso(alpha=0.1),
}

for name, model in models.items():
    model.fit(X_train_scaled ,Y_train)
    y_predict=model.predict(X_test_scaled)
    
    MSE=mean_squared_error(Y_test,y_predict)
    MAE=mean_absolute_error(Y_test,y_predict)
    rs=r2_score(Y_test,y_predict)
    
    score[name] = {"Mean Squared error":MSE,"Mean Absolute error":MAE,"R2 Score":rs}
    
    


# In[81]:


score


# In[84]:


import joblib

joblib.dump(model_reg,"weader_model_reg.pkl")
joblib.dump(model_lasso,"weader_model_lasso.pkl")
joblib.dump(model_ridge,"weader_model_ridge.pkl")
joblib.dump(scaler,"weather_scaler.pkl")


# In[ ]:





# In[85]:


location_mapping = {
    "New York": 3,
    "London": 2,
    "Tokyo": 7,
    "Paris": 4,
    "Sydney": 6,
    "Dubai": 0,
    "Rome": 5,
    "Hong Kong": 1
}
def get_location_index(location_name):
    return location_mapping[location_name]


location_index = get_location_index("Tokyo") 
location_index


# In[86]: