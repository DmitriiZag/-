import pandas as pd
import sqlite3
import numpy as np
import os


set_5 = pd.read_excel('/Users/dmitrii/Downloads/10. ДЖКХ + ДИТ 2/5. Перечень событий за период 01.10.2023-30.04.2023 (ЦУ КГХ)/All_events.xlsx')
set_6 = pd.read_excel('/Users/dmitrii/Downloads/10. ДЖКХ + ДИТ 2/6. Плановые-Внеплановые отключения 01.10.2023-30.04.2023.xlsx')
set_9 = pd.read_excel('/Users/dmitrii/Downloads/10. ДЖКХ + ДИТ 2/9. Выгрузка БТИ.xlsx')
set_11 = pd.read_excel('/Users/dmitrii/Downloads/10. ДЖКХ + ДИТ 2/11.Выгрузка_ОДПУ_отопление_ВАО_20240522.xlsx')
set_14 = pd.read_excel('/Users/dmitrii/Downloads/10. ДЖКХ + ДИТ 2/14. ВАО_Многоквартирные_дома_с_технико_экономическими_характеристиками.xlsx')


# Define the path where the pickle files will be stored
output_path = '/Users/dmitrii/Desktop/Хакатон/Datasets'

# Check if the directory exists, if not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save each DataFrame as a pickle file
set_5.to_pickle(os.path.join(output_path, 'Set 5.pkl'))
set_6.to_pickle(os.path.join(output_path, 'Set 6.pkl'))
set_9.to_pickle(os.path.join(output_path, 'Set 9.pkl'))
set_11.to_pickle(os.path.join(output_path, 'Set 11.pkl'))
set_14.to_pickle(os.path.join(output_path, 'Set 14.pkl'))

