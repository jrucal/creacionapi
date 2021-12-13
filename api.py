#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Binarizer

from sklearn.pipeline import Pipeline

from feature_engine.imputation import(
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder
)

from feature_engine.transformation import LogTransformer

from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

import joblib


# In[3]:


import my_preprocessors as mypp


# In[7]:


data = pd.read_csv("titanic_train.csv")
data.head()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
                        data.drop(["PassengerId","Age"], axis=1 ),
                        data["Age"],
                        test_size=0.15, 
                        random_state=2021)


# In[11]:


X_train.shape, X_test.shape


# In[12]:


y_train = np.log(y_train)
y_test = np.log(y_test)


# In[13]:


categorica = [var for var in data.columns if data[var].dtype =="O"]
categorica = categorica + ["Pclass"]


# In[14]:


X_train[categorica] = X_train[categorica].astype("O")
X_test[categorica]= X_test[categorica].astype("O")


# In[15]:


cat_with_na = [var for var in categorica
              if X_train[var].isnull().sum()>0]


# In[16]:


cat_with_na


# In[18]:


X_train[cat_with_na].isnull().mean().sort_values(ascending = False)


# In[19]:


vars_with_missing_string = [var for var in cat_with_na
                           if X_train[var].isnull().mean()>0.2]


# In[20]:


vars_freq_category = [var for var in cat_with_na
                           if X_train[var].isnull().mean()<=0.2]


# In[21]:


X_train[vars_with_missing_string] = X_train[vars_with_missing_string].fillna("Missing")
X_test[vars_with_missing_string] = X_test[vars_with_missing_string].fillna("Missing")


# In[24]:


for var in vars_freq_category:
    mode=X_train[var].mode()[0]
    
    X_train[var].fillna(mode, inplace=True)
    X_test[var].fillna(mode, inplace=True)
    
    print(var, "____", mode)


# In[27]:


X_train[cat_with_na].isnull().mean().sort_values(ascending = False)


# In[28]:


cat_with_na = [var for var in categorica
              if X_train[var].isnull().sum()>0]

cat_with_na


# In[29]:


num_vars = [var for var in X_train.columns
              if var not in categorica and var !="Age"]


# In[30]:


len(num_vars)


# In[31]:


nums_with_na = [var for var in num_vars
              if X_train[var].isnull().sum()>0]


# In[32]:


nums_with_na


# In[33]:


X_train.to_csv("preprocess_data/prep_Xtrain.csv", index=False)


# In[34]:


X_test.to_csv("preprocess_data/prep_Xtest.csv", index=False)


# In[35]:


y_train.to_csv("preprocess_data/prep_ytrain", index=False)


# In[36]:


y_test.to_csv("preprocess_data/prep_ytest", index=False)


# In[37]:


X_train = pd.read_csv("preprocess_data/prep_Xtrain.csv")
X_test = pd.read_csv("preprocess_data/prep_Xtest.csv")


# In[38]:


X_train.head()


# In[39]:


sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=0))


# ## Configuración del Machine Learning Pipeline

# In[42]:



#Varibles para transformación logaritmia
NUMERICALS_LOG_VARS = ["Age"]

#Variables para hacer mapeo categorico por codificación ordinal
QUAL_VARS = ["Sex", "Embarked"]

#Variables categoricas a codificar sin ordinalidad
CATEGORICAL_VARS = ["Name", "Ticket","Cabin","Pclass"]

#Mapeos de variables categoricas
quality_mapping = {"female":1, "male":2, "NaN":0, "S":3, "C":4}

#Variables seleccionadas según análisis de Lasso
FEATURES = ["Name", "Ticket","Cabin","Pclass", "Sex", "Embarked"
    
]


# In[43]:


X_train = X_train[FEATURES]


# ## Machine Learing PipeLine

# In[44]:


Age_pipeline = Pipeline([
    
    # Tratamiento de variables temporales
    ('eslapsed_time', mypp.TremporalVariableTransformer(
        variables=TEMPORAL_VARS, reference_variable=REF_VAR)
    ),
    
    #============= TRANSFORMACIÓN DE VARIABLES NUMÉRICAS =============
    
    # Transformación logaritmica
    ('log', LogTransformer(variables=NUMERICALS_LOG_VARS)),
    
    
    #=============== CODIFICACION DE VARIABLES CATEGORICAS ORDINALES ==============
    ('mapper_quality', mypp.Mapper(
        variables=QUAL_VARS, mappings=QUAL_MAPPINGS)),
    
    #============ CODIFICACION DE VARIABLES CATEGORICAS NOMINALES ============
    
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.01, n_categories=1, variables=CATEGORICAL_VARS)),
    
    ('categorical_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=CATEGORICAL_VARS)),
    
    #=========== SCALER ==============
    ('scaler', MinMaxScaler()),
    
    #=========== ENTRENAMIENTO DEL MODELO ============
    ('Lasso', Lasso(alpha=0.01, random_state=2022)),
]) 


# In[ ]:


Age_pipeline.fit(X_train, y_train)


# In[ ]:


#Cargamos dataset test.csv para prueba.
X_test = pd.read_csv("test.csv")
X_test = X_test[FEATURES]


# In[ ]:


preds = Age_pipeline.predict(X_test)
preds


# In[ ]:


from sklearn.metrics import mean_squared_error 


# In[ ]:


mean_squared_error(np.exp(y_test), np.exp(preds), squared=False)


# In[ ]:


X_test


# In[ ]:


import joblib


# In[ ]:


#Guardamos pipeline
joblib.dump(Age_pipeline, 'Age_pipeline.pkl')


# In[ ]:


type(Age_pipeline)

