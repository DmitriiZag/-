import pandas as pd
import numpy as np
import os

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
print('done')
set_6 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 6.pkl')
print('done')
set_9 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 9.pkl')
print('done')
set_11 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 11.pkl')
print('done')
set_14 = pd.read_pickle('/Users/dmitrii/Desktop/Хакатон/Datasets/Set 14.pkl')
print('done')

# Step 1: Shuffle set_5 and select 10%
set_5_shuffled = set_5.sample(frac=1, random_state=1)  # Shuffle
subset_set_5 = set_5_shuffled.head(int(0.1 * len(set_5)))  # Take 10%
print('done')

result = subset_set_5.copy()  # Start with the 10% subset of set_5
result = pd.merge(result, set_6, on='УНОМ', how='left', suffixes=('', '_set6'))
result = pd.merge(result, set_9, on='УНОМ', how='left', suffixes=('', '_set9'))
#result = pd.merge(result, set_11, on='УНОМ', how='left', suffixes=('', '_set11'))
result = pd.merge(result, set_14, on='УНОМ', how='left', suffixes=('', '_set14'))
print('done')



