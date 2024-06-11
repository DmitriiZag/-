#%% md
# Jupyter notebook sample
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
#%%
# Load datasets
set_5 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 5.pkl')
set_9 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 9.pkl')
set_11 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 11.pkl')
set_14 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 14.pkl')
#%%
# Define the conditions you're interested in
conditions = [
    "P1 <= 0",
    "P2 <= 0",
    "T1 < min",
    "T1 > max",
    "Протечка труб в подъезде",
    "Сильная течь в системе отопления",
    "Температура в квартире ниже нормативной",
    "Температура в помещении общего пользования ниже нормативной",
    "Течь в системе отопления"
]
#%%
# Filter set_5 to include only rows with 'Наименование' matching any of the conditions
set_5 = set_5[set_5['Наименование'].isin(conditions)]
#%%

#%%

#%%
# Shuffle set_5 and select 100%
set_5_shuffled = set_5.sample(frac=1, random_state=1)
subset_set_5 = set_5_shuffled
#%%
# Convert specified columns to numeric, setting errors='coerce'
columns_to_convert = [
    'Объём поданого теплоносителя в систему ЦО',
    'Объём обратного теплоносителя из системы ЦО',
    'Разница между подачей и обраткой(Подмес)',
    'Разница между подачей и обраткой(Утечка)',
    'Температура подачи',
    'Температура обратки',
    'Наработка часов счётчика',
    'Расход тепловой энергии '
]
for column in columns_to_convert:
    set_11[column] = pd.to_numeric(set_11[column], errors='coerce')
#%%
# Perform aggregation
set_11 = set_11.groupby('УНОМ').agg({
    'Объём поданого теплоносителя в систему ЦО': 'mean',
    'Объём обратного теплоносителя из системы ЦО': 'mean',
    'Разница между подачей и обраткой(Подмес)': 'mean',
    'Разница между подачей и обраткой(Утечка)': 'mean',
    'Температура подачи': 'mean',
    'Температура обратки': 'mean',
    'Наработка часов счётчика': 'mean',
    'Расход тепловой энергии ': 'mean'
}).reset_index()
#%%
# Merge datasets
result = subset_set_5.copy()
result = pd.merge(result, set_9, on='УНОМ', how='left')
result = pd.merge(result, set_11, on='УНОМ', how='left')
result = pd.merge(result, set_14, on='УНОМ', how='left')
print(result.columns)
#%%
# Drop irrelevant columns
drop_columns = [
    'Тип номера дом', 'Тип номера строения/сооружения', 'Тип', 'Признак', 'Идентификатор из сторонней системы',
    'Общая площадь_x', 'Unnamed: 16', 'Адрес', 'Общая площадь нежилых помещений',
    'Дата создания во внешней системе', 'Дата закрытия', 'Очередность уборки кровли', 'Типы жилищного фонда',
    'Количество грузопассажирских лифтов'
]
result.drop(columns=drop_columns, inplace=True)
#%%
# Fill NaN for numerical and propagate for categorical
numerical_columns = result.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = result.select_dtypes(exclude=['float64', 'int64', 'int32', 'float32']).columns

for column in numerical_columns:
    mean_value = result[column].mean()
    result[column].fillna(mean_value, inplace=True)

for column in categorical_columns:
    result[column] = result.groupby('УНОМ')[column].transform(lambda x: x.ffill().bfill())
    if result[column].isnull().any():
        most_common = result[column].mode()[0] if not result[column].mode().empty else "Unknown"
        result[column].fillna(most_common, inplace=True)
#%%
# Separate label encoder for the target variable
le_target = LabelEncoder()
result['Наименование'] = le_target.fit_transform(result['Наименование'])

# Separate label encoder for categorical features
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    result[column] = le.fit_transform(result[column])
    label_encoders[column] = le
#%%
# Prepare data for model
X = result.drop('Наименование', axis=1)
y = result['Наименование']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Model training
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
#%%
# Prediction and evaluation
test_x = X_test.drop_duplicates(subset=['УНОМ']).copy()
test_y = y_test.loc[test_x.index]
y_pred = rf_classifier.predict(test_x)

test_x['Predictions'] = y_pred
test_x['Emergency_prediction'] = le_target.inverse_transform(test_x['Predictions'])

accuracy = accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(test_y, y_pred))
#%%
# Decode categorical features
for column in categorical_columns:
    if column in test_x.columns:
        le = label_encoders[column]
        test_x[column] = le.inverse_transform(test_x[column])
    else:
        print(f"Column {column} not found in test_x")

#%%
# Save the DataFrame to CSV
file_path = '/Users/dmitrii/Desktop/Хакатон/Datasets/predictions.csv'
test_x.to_csv(file_path, index=False)

print(f"Predictions saved to {file_path}")