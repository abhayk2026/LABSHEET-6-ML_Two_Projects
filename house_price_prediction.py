import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('house_prices.csv')
X = df.drop(columns=['Price'])
y = df['Price']
cat = ['Location']
num = ['Area', 'Bedrooms', 'Bathrooms', 'Parking', 'Age']
preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat)], remainder='passthrough')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(n_estimators=200, random_state=42)
}
for name, estimator in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print('\n', name)
    print('MAE:', mean_absolute_error(y_test, pred))
    print('RMSE:', mean_squared_error(y_test, pred) ** 0.5)
    print('R2 Score:', r2_score(y_test, pred))

best_model = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=42))])
best_model.fit(X_train, y_train)
new_house = pd.DataFrame({'Area':[1800], 'Bedrooms':[3], 'Location':['City Center'], 'Bathrooms':[2], 'Parking':[1], 'Age':[5]})
print('\nPredicted price for new house:', best_model.predict(new_house)[0])
joblib.dump(best_model, 'house_price_model.joblib')
print('Model saved as house_price_model.joblib')
