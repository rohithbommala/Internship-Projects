#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering and Modelling
# 
# ---
# 
# 1. Import packages
# 2. Load data
# 3. Modelling
# 
# ---
# 
# ## 1. Import packages

# In[2]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# Shows plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style
sns.set(color_codes=True)


# ---
# ## 2. Load data

# In[4]:


df = pd.read_csv('data_for_predictions')
df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()


# ---
# 
# ## 3. Modelling
# 
# We now have a dataset containing features that we have engineered and we are ready to start training a predictive model. Remember, we only need to focus on training a `Random Forest` classifier.

# In[5]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ### Data sampling
# 
# The first thing we want to do is split our dataset into training and test samples. The reason why we do this, is so that we can simulate a real life situation by generating predictions for our test sample, without showing the predictive model these data points. This gives us the ability to see how well our model is able to generalise to new data, which is critical.
# 
# A typical % to dedicate to testing is between 20-30, for this example we will use a 75-25% split between train and test respectively.

# In[6]:


# Make a copy of our data
train_df = df.copy()

# Separate target variable from independent variables
y = df['churn']
X = df.drop(columns=['id', 'churn'])
print(X.shape)
print(y.shape)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Model training
# 
# Once again, we are using a `Random Forest` classifier in this example. A Random Forest sits within the category of `ensemble` algorithms because internally the `Forest` refers to a collection of `Decision Trees` which are tree-based learning algorithms. As the data scientist, you can control how large the forest is (that is, how many decision trees you want to include).
# 
# The reason why an `ensemble` algorithm is powerful is because of the laws of averaging, weak learners and the central limit theorem. If we take a single decision tree and give it a sample of data and some parameters, it will learn patterns from the data. It may be overfit or it may be underfit, but that is now our only hope, that single algorithm. 
# 
# With `ensemble` methods, instead of banking on 1 single trained model, we can train 1000's of decision trees, all using different splits of the data and learning different patterns. It would be like asking 1000 people to all learn how to code. You would end up with 1000 people with different answers, methods and styles! The weak learner notion applies here too, it has been found that if you train your learners not to overfit, but to learn weak patterns within the data and you have a lot of these weak learners, together they come together to form a highly predictive pool of knowledge! This is a real life application of many brains are better than 1.
# 
# Now instead of relying on 1 single decision tree for prediction, the random forest puts it to the overall views of the entire collection of decision trees. Some ensemble algorithms using a voting approach to decide which prediction is best, others using averaging. 
# 
# As we increase the number of learners, the idea is that the random forest's performance should converge to its best possible solution.
# 
# Some additional advantages of the random forest classifier include:
# 
# - The random forest uses a rule-based approach instead of a distance calculation and so features do not need to be scaled
# - It is able to handle non-linear parameters better than linear based models
# 
# On the flip side, some disadvantages of the random forest classifier include:
# 
# - The computational power needed to train a random forest on a large dataset is high, since we need to build a whole ensemble of estimators.
# - Training time can be longer due to the increased complexity and size of thee ensemble

# In[8]:


model = RandomForestClassifier(
    n_estimators=1000
)
model.fit(X_train, y_train)


# The `scikit-learn` documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>, has a lot of information about the algorithm and the parameters that you can use when training a model.
# 
# For this example, I am using `n_estimators` = 1000. This means that my random forest will consist of 1000 decision trees. There are many more parameters that you can fine-tune within the random forest and finding the optimal combinations of parameters can be a manual task of exploration, trial and error, which will not be covered during this notebook.

# ### Evaluation
# 
# Now let's evaluate how well this trained model is able to predict the values of the test dataset.
# 
# We are going to use 3 metrics to evaluate performance:
# 
# - Accuracy = the ratio of correctly predicted observations to the total observations
# - Precision = the ability of the classifier to not label a negative sample as positive
# - Recall = the ability of the classifier to find all the positive samples
# 
# The reason why we are using these three metrics is because a simple accuracy is not always a good measure to use. To give an example, let's say you're predicting heart failures with patients in a hospital and there were 100 patients out of 1000 that did have a heart failure. 
# 
# If you predicted 80 out of 100 (80%) of the patients that did have a heart failure correctly, you might think that you've done well! However, this also means that you predicted 20 wrong and what may the implications of predicting these remaining 20 patients wrong? Maybe they miss out on getting vital treatment to save their lives. 
# 
# As well as this, what about the impact of predicting negative cases as positive (people not having heart failure being predicted that they did), maybe a high number of false positives means that resources get used up on thee wrong people and a lot of time is wasted when they could have been helping the real heart failure sufferers. 
# 
# This is just an example, but it illustrates why other performance metrics are necessary such `Precision` and `Recall`, which are good measures to use in a classification scenario.

# In[9]:


predictions = model.predict(X_test)
tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()


# In[10]:


y_test.value_counts()


# In[11]:


print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}\n")

print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
print(f"Precision: {metrics.precision_score(y_test, predictions)}")
print(f"Recall: {metrics.recall_score(y_test, predictions)}")


# Looking at these results there are a few things to point out:
# 
# <b>Note:</b> If you are running this notebook yourself, you may get slightly different answers!
# 
# - Within the test set about 10% of the rows are churners (churn = 1).
# - Looking at the true negatives, we have 3282 out of 3286. This means that out of all the negative cases (churn = 0), we predicted 3282 as negative (hence the name True negative). This is great!
# - Looking at the false negatives, this is where we have predicted a client to not churn (churn = 0) when in fact they did churn (churn = 1). This number is quite high at 348, we want to get the false negatives to as close to 0 as we can, so this would need to be addressed when improving the model.
# - Looking at false positives, this is where we have predicted a client to churn when they actually didnt churn. For this value we can see there are 4 cases, which is great!
# - With the true positives, we can see that in total we have 366 clients that churned in the test dataset. However, we are only able to correctly identify 18 of those 366, which is very poor.
# - Looking at the accuracy score, this is very misleading! Hence the use of precision and recall is important. The accuracy score is high, but it does not tell us the whole story.
# - Looking at the precision score, this shows us a score of 0.82 which is not bad, but could be improved.
# - However, the recall shows us that the classifier has a very poor ability to identify positive samples. This would be the main concern for improving this model!
# 
# So overall, we're able to very accurately identify clients that do not churn, but we are not able to predict cases where clients do churn! What we are seeing is that a high % of clients are being identified as not churning when they should be identified as churning. This in turn tells me that the current set of features are not discriminative enough to clearly distinguish between churners and non-churners. 
# 
# A data scientist at this point would go back to feature engineering to try and create more predictive features. They may also experiment with optimising the parameters within the model to improve performance. For now, lets dive into understanding the model a little more.
# 
# ### Model understanding
# 
# A simple way of understanding the results of a model is to look at feature importances. Feature importances indicate the importance of a feature within the predictive model, there are several ways to calculate feature importance, but with the Random Forest classifier, we're able to extract feature importances using the built-in method on the trained model. In the Random Forest case, the feature importance represents the number of times each feature is used for splitting across all trees.

# In[39]:


feature_importances = pd.DataFrame({
    'features': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=True).reset_index()


# In[40]:


plt.figure(figsize=(15, 25))
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances['importance'], color='b', align='center')
plt.yticks(range(len(feature_importances)), feature_importances['features'])
plt.xlabel('Importance')
plt.show()


# From this chart, we can observe the following points:
# 
# - Net margin and consumption over 12 months is a top driver for churn in this model
# - Margin on power subscription also is an influential driver
# - Time seems to be an influential factor, especially the number of months they have been active, their tenure and the number of months since they updated their contract
# - The feature that our colleague recommended is in the top half in terms of how influential it is and some of the features built off the back of this actually outperform it
# - Our price sensitivity features are scattered around but are not the main driver for a customer churning
# 
# The last observation is important because this relates back to our original hypothesis:
# 
#     > Is churn driven by the customers' price sensitivity?
# 
# Based on the output of the feature importances, it is not a main driver but it is a weak contributor. However, to arrive at a conclusive result, more experimentation is needed.

# In[41]:


proba_predictions = model.predict_proba(X_test)
probabilities = proba_predictions[:, 1]


# In[42]:


X_test = X_test.reset_index()
X_test.drop(columns='index', inplace=True)


# In[43]:


X_test['churn'] = predictions.tolist()
X_test['churn_probability'] = probabilities.tolist()
X_test.to_csv('out_of_sample_data_with_predictions.csv')

