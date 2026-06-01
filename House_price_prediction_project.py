# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# %%
hp=pd.read_csv("house_prices.csv")

# %%
hp.head()

# %%
hp.info()
hp.shape

# %%
hp.isnull().sum()

# %%
hp.describe()

# %%

plt.figure(figsize=(12,8))
sns.heatmap(hp.corr(numeric_only=True), cmap='coolwarm')
plt.show()
sns.histplot(hp['price'], bins=30)
plt.show()
cols = ['price','sqft_living','bathrooms','grade']
sns.pairplot(hp[cols])
plt.show()

# %%
hp['renovated'] = (hp['yr_renovated'] > 0).astype(int)



# %%
hp=hp.drop('yr_renovated',axis=1)

# %%
hp['date'] = pd.to_datetime(hp['date'])

# %%
hp['sale_year'] = hp['date'].dt.year
hp['sale_month'] = hp['date'].dt.month

# %%
hp.drop("date",axis=1,inplace=True)

# %%
hp = pd.get_dummies(
    hp,
    columns=['waterfront','condition'],
    drop_first=True
)

# %%
hp

# %%
from sklearn.model_selection import train_test_split
X=hp[["bedrooms","bathrooms","sqft_living","sqft_lot","floors",	"view",	"grade"	,"sqft_above","sqft_basement",	"house_age",	"renovated",	"zipcode",	"lat",	"long",	"sqft_living15",	"sqft_lot15","sale_year","sale_month","waterfront_Y"	,"condition_Fair",	"condition_Good",	'condition_Poor',	"condition_Very Good"]]
y=hp["price"]


# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# %%
lm=LinearRegression()
lm.fit(X_train_scaled,y_train)


# %%
ridge=Ridge(alpha=1)
ridge.fit(X_train_scaled,y_train)

# %%
lasso=Lasso(alpha=1,max_iter=50000)
lasso.fit(X_train_scaled,y_train)

# %%
pred_linear = lm.predict(X_test_scaled)

pred_ridge = ridge.predict(X_test_scaled)

pred_lasso = lasso.predict(X_test_scaled)

# %%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

# %%
print("Linear Regression")

print("MAE:",mean_absolute_error(y_test,pred_linear))

print("RMSE:",
      math.sqrt(mean_squared_error(y_test,pred_linear)))

print("R2:",
      r2_score(y_test,pred_linear))

# %%
print("Ridge")

print("MAE:",mean_absolute_error(y_test,pred_ridge))

print("RMSE:",
      math.sqrt(mean_squared_error(y_test,pred_ridge)))

print("R2:",
      r2_score(y_test,pred_ridge))

# %%
print("Lasso")

print("MAE:",mean_absolute_error(y_test,pred_lasso))

print("RMSE:",
      math.sqrt(mean_squared_error(y_test,pred_lasso)))

print("R2:",
      r2_score(y_test,pred_lasso))

# %%
residuals = y_test - pred_lasso

# %%
sns.histplot(residuals,kde=True)
plt.show()

# %%
best_pred = pred_linear

residuals = y_test - best_pred

sns.histplot(residuals, kde=True)
plt.show()

# %%
sns.scatterplot(x=y_test, y=best_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()

# %%
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lm.coef_
})

coef_df.sort_values(
    by="Coefficient",
    ascending=False
).head(10)

# %%



