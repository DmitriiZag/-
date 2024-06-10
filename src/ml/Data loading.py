import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load datasets from specified paths."""
    data_paths = {
        'set_5': '/Users/dmitrii/Desktop/Хакатон/Datasets/Set 5.pkl',
        'set_9': '/Users/dmitrii/Desktop/Хакатон/Datasets/Set 9.pkl',
        'set_11': '/Users/dmitrii/Desktop/Хакатон/Datasets/Set 11.pkl',
        'set_14': '/Users/dmitrii/Desktop/Хакатон/Datasets/Set 14.pkl'
    }
    return {name: pd.read_pickle(path) for name, path in data_paths.items()}

def preprocess_data(data):
    """Convert specific columns to numeric and handle NaNs."""
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
        data['set_11'][column] = pd.to_numeric(data['set_11'][column], errors='coerce')
    data['set_11'] = data['set_11'].groupby('УНОМ').agg({col: 'mean' for col in columns_to_convert}).reset_index()

def merge_datasets(data):
    """Merge datasets into a single DataFrame after preprocessing."""
    result = data['set_5'].copy()
    for key in ['set_9', 'set_11', 'set_14']:
        result = pd.merge(result, data[key], on='УНОМ', how='left')
    return result

def clean_data(df):
    """Drop irrelevant columns and fill NaN values."""
    drop_columns = [
        'Тип номера дом', 'Тип номера строения/сооружения', 'Тип', 'Признак',
        'Идентификатор из сторонней системы', 'Общая площадь_x', 'Unnamed: 16',
        'Адрес', 'Общая площадь нежилых помещений', 'Дата создания во внешней системе',
        'Дата закрытия', 'Очередность уборки кровли', 'Типы жилищного фонда', 'Количество грузопассажирских лифтов'
    ]
    df.drop(columns=drop_columns, inplace=True)
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column].fillna(df[column].mean(), inplace=True)
    for column in df.select_dtypes(exclude=['float64', 'int64', 'int32', 'float32']).columns:
        df[column] = df.groupby('УНОМ')[column].transform(lambda x: x.ffill().bfill())
        if df[column].isnull().any():
            df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else "Unknown", inplace=True)

def encode_features(df):
    """Encode categorical features and target variable."""
    le_target = LabelEncoder()
    df['Наименование'] = le_target.fit_transform(df['Наименование'])
    for column in df.select_dtypes(include=['category', 'object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    return le_target

def train_model(X, y):
    """Train a RandomForestClassifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, le_target):
    """Evaluate the model and return the test set with predictions."""
    test_x = X_test.drop_duplicates(subset=['УНОМ']).copy()
    test_y = y_test.loc[test_x.index]
    y_pred = model.predict(test_x)
    test_x['Predictions'] = y_pred
    test_x['Predicted_Labels'] = le_target.inverse_transform(y_pred)
    print("Accuracy:", accuracy_score(test_y, y_pred))
    print("Classification Report:\n", classification_report(test_y, y_pred))
    return test_x

def save_predictions(df):
    """Save DataFrame with predictions to CSV."""
    file_path = '/Users/dmitrii/Desktop/Хакатон/Datasets/predictions.csv'
    df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def main():
    data = load_data()
    preprocess_data(data)
    result = merge_datasets(data)
    clean_data(result)
    le_target = encode_features(result)
    X = result.drop('Наименование', axis=1)
    y = result['Наименование']
    model, X_test, y_test = train_model(X, y)
    test_x = evaluate_model(model, X_test, y_test, le_target)
    save_predictions(test_x)

if __name__ == "__main__":
    main()
