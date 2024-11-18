import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


data_df = pd.read_csv('C:/Users/shafiee/Desktop/Plant_1_Generation_Data.csv')

#print(data_df.info())
#print(data_df.duplicated())
#print(data_df.isnull().sum())
#print(data_df.shape)

data_df = data_df.dropna().reset_index(drop=True)
data_df.drop(['SOURCE_KEY'], axis=1, inplace=True)
data_df.drop(['DATE_TIME'], axis=1, inplace=True)

for column in ['PLANT_ID', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']:
    data_df[column] = pd.to_numeric(data_df[column], errors='coerce')

X = data_df.iloc[: , 1:5]
y = data_df['TOTAL_YIELD']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=68)


sample_size = int(0.1 * X_train.shape[0])
X_train = X_train[:sample_size]  
y_train = y_train.iloc[:sample_size]

models ={
    "Linear Regression": LinearRegression(),
    "Decision Tree ": DecisionTreeRegressor(random_state=42)
    }

for models_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_score = mean_absolute_error(y_test, y_pred)
    mse_score = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{models_name}:")
    print(f'MAE: {mae_score}')
    print(f'MSE: {mse_score}')
    print(f'R: {r2}')


