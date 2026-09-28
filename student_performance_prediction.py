import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('student_performance.csv')
X = df[['Attendance', 'InternalMarks', 'Assignments', 'StudyHours', 'PreviousSemesterResult']]
y = df['FinalScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, pred))
print('RMSE:', mean_squared_error(y_test, pred) ** 0.5)
print('R2 Score:', r2_score(y_test, pred))
new_student = pd.DataFrame({'Attendance':[90], 'InternalMarks':[82], 'Assignments':[88], 'StudyHours':[6], 'PreviousSemesterResult':[78]})
print('Predicted Final Score:', model.predict(new_student)[0])
joblib.dump(model, 'student_performance_model.joblib')
print('Model saved as student_performance_model.joblib')
