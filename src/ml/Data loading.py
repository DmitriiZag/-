import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

'''
# Full path to the Excel file
file_path = '/Users/dmitrii/Downloads/10. ДЖКХ + ДИТ 2/11.Выгрузка_ОДПУ_отопление_ВАО_20240522.xlsx'
pickle_file_path = '/Users/dmitrii/Desktop/Хакатон/Datasets/Set 11.pkl'

# Check if the pickle file exists
if os.path.isfile(pickle_file_path):
    print(f"Loading data from pickle file: {pickle_file_path}")
    people = pd.read_pickle(pickle_file_path)
else:
    print(f"Reading data from Excel file: {file_path}")
    people = pd.read_excel(file_path)
    # Save the DataFrame as a pickle file for future use
    people.to_pickle(pickle_file_path)
'''

set_5 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 5.pkl')
set_9 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 9.pkl')
set_11 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 11.pkl')
set_14 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 14.pkl')


# Step 1: Shuffle set_5 and select 10%
set_5_shuffled = set_5.sample(frac=1, random_state=1)  # Shuffle
subset_set_5 = set_5_shuffled.head(int(0.3 * len(set_5)))  # Take 10%


# Convert columns to numeric, setting errors='coerce' will convert non-numeric values to NaN
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

# Now perform the aggregation
set_11 = set_11.groupby('УНОМ').agg({
    'Объём поданого теплоносителя в систему ЦО': 'mean',
    'Объём обратного теплоносителя из системы ЦО': 'mean',
    'Разница между подачей и обраткой(Подмес)': 'mean',
    'Разница между подачей и обраткой(Утечка)': 'mean',
    'Температура подачи': 'mean',
    'Температура обратки': 'mean',
    'Наработка часов счётчика': 'mean',
    'Расход тепловой энергии ': 'mean'  # Make sure to use correct column name
}).reset_index()


result = subset_set_5.copy()  # Start with the 10% subset of set_5
result = pd.merge(result, set_9, on='УНОМ', how='left', suffixes=('', '_set9'))
result = pd.merge(result, set_11, on='УНОМ', how='left', suffixes=('', '_set11'))
result = pd.merge(result, set_14, on='УНОМ', how='left', suffixes=('', '_set14'))

result = result.drop('Тип номера дом', axis=1)
result = result.drop('Тип номера строения/сооружения', axis=1)
result = result.drop('Тип', axis=1)
result = result.drop('Признак', axis=1)
result = result.drop('Идентификатор из сторонней системы', axis=1)
result = result.drop('Общая площадь_set14', axis=1)
result = result.drop('Unnamed: 16', axis=1)
result = result.drop('Адрес', axis=1)
result = result.drop('Общая площадь нежилых помещений', axis=1)
result = result.drop('Дата создания во внешней системе', axis=1)
result = result.drop('Дата закрытия', axis=1)
result = result.drop('Очередность уборки кровли', axis=1)
result = result.drop('Типы жилищного фонда', axis=1)
result = result.drop('Количество грузопассажирских лифтов', axis=1)



result['Наименование'] = result['Наименование'].astype('category')
result['УНОМ'] = result['УНОМ'].astype('category')
result['Материал'] = result['Материал'].astype('category')
result['Назначение'] = result['Назначение'].astype('category')
result['Класс'] = result['Класс'].astype('category')
result['Этажность'] = result['Этажность'].astype('float')
result['Общая площадь'] = result['Общая площадь'].str.replace(',', '.')
result['Общая площадь'] = result['Общая площадь'].astype(float)
result['Серии проектов'] = result['Серии проектов'].astype('category')
result[' Материалы стен'] = result[' Материалы стен'].astype('category')
result['Признак аварийности здания'] = result['Признак аварийности здания'].astype('category')
result['Материалы кровли по БТИ'] = result['Материалы кровли по БТИ'].astype('category')
result['Статусы МКД'] = result['Статусы МКД'].astype('category')





categorical_columns = ['Наименование', 'УНОМ', 'Материал', 'Назначение', 'Класс', 'Серии проектов',
                       ' Материалы стен', 'Признак аварийности здания',
                       'Материалы кровли по БТИ', 'Статусы МКД']

le = LabelEncoder()

for column in categorical_columns:
    result[column] = le.fit_transform(result[column])





numerical_columns = result.select_dtypes(include=['float64', 'int64']).columns

# Step 2: Replace NaN values with the mean of each column
for column in numerical_columns:
    mean_value = result[column].mean()  # Calculate the mean
    result[column].fillna(mean_value, inplace=True)  # Replace NaNs

# Optional: Check if there are any NaNs left
nan_counts_after = result.isna().sum()
print(nan_counts_after)





X = result.drop('Наименование', axis=1)
y = result['Наименование']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

