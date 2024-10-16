import os
import glob
import pandas as pd
from pathlib import Path
from utils.logger import info, warning
import re

def load_pud_data(config):
    """
    Загружает данные ПУДов из указанных Excel файлов и объединяет их в один DataFrame.

    :param config: Конфигурационный словарь.
    :return: Очищенный DataFrame с ПУДами.
    """
    path_to_files = config['data_path']

    # Чтение всех файлов Excel и объединение их в один DataFrame
    xlsx_files = glob.glob(os.path.join(path_to_files, "*.xlsx"))
    df = pd.DataFrame()

    for file in xlsx_files:
        df_temp = pd.read_excel(file)
        df = pd.concat([df, df_temp], ignore_index=True)

    # Проверка наличия столбца с ID дисциплины
    if 'ID дисциплины БУП ППК (АСАВ)' not in df.columns:
        raise ValueError("Столбец 'ID дисциплины БУП ППК (АСАВ)' не найден в данных.")

    # Извлечение года из периода изучения дисциплины и замена NaN на '0000/0000'
    df['year'] = df['Период изучения дисциплины'].fillna('').astype(str).str.findall(r'(\d{4}/\d{4})').apply(lambda x: x[-1] if x else '0000/0000')

    # Проверка наличия строк без года изучения (значение '0000/0000')
    missing_years = df[df['year'] == '0000/0000']
    if not missing_years.empty:
        warning(f"Обнаружены строки без информации о периоде изучения: {len(missing_years)} записей.")

    # Сортировка по 'Русскоязычное название дисциплины' и 'year', чтобы оставить самые актуальные дисциплины
    df_sorted = df.sort_values(by=['Русскоязычное название дисциплины', 'year'], ascending=[True, False])

    # Удаление дубликатов, сохраняя только самые актуальные дисциплины (с последним периодом изучения)
    df_cleaned = df_sorted.drop_duplicates(subset=['Русскоязычное название дисциплины', 'Факультет кафедры, предлагающей дисциплину'], keep='first')

    # Формирование столбца Full_Info для объединения текстовых столбцов
    df_cleaned.loc[:, 'Full_Info'] = (
        df_cleaned['Русскоязычное название дисциплины'] +
        '\nАннотация: ' + df_cleaned['Аннотация'].fillna('') +
        '\nСписок разделов: ' + df_cleaned['Список разделов (названия и описания)'].fillna('') +
        '\nСписок планируемых результатов обучения: ' + df_cleaned['Список планируемых результатов обучения РПУДа'].fillna('')
    )

    return df_cleaned


def filter_rows_by_mode(df, config):
    """
    Фильтрует строки DataFrame в зависимости от режима обработки, заданного в конфигурации.

    :param df: DataFrame с данными.
    :param config: Конфигурационный словарь с параметрами обработки.
    :return: DataFrame с отфильтрованными строками.
    """
    mode = config['processing']['mode']

    if mode == 'all':
        # Возвращаем весь DataFrame, если выбран режим 'all'
        return df

    elif mode == 'solo':
        # Обрабатываем только одну строку с указанным индексом
        solo_index = config['processing']['solo_index']
        info(f"Обработка единственной строки с индексом {solo_index}.")
        return df.iloc[[solo_index]]

    elif mode == 'random':
        # Обрабатываем строки в случайном порядке
        info("Обработка всех строк случайным образом.")
        return df.sample(frac=1)

    elif mode == 'discipline_name':
        # Фильтруем по названию дисциплины
        discipline_name = config['processing']['discipline_name']
        info(f"Фильтрация строк по названию дисциплины: {discipline_name}")
        filtered_df = df[df['Русскоязычное название дисциплины'].str.contains(discipline_name, case=False, na=False)]

        if filtered_df.empty:
            warning(f"Не найдено строк с названием дисциплины: {discipline_name}")
        return filtered_df

    elif mode == 'id_search':
        # Поиск по ID или списку ID
        ids_to_search = config['processing'].get('id_list', [])

        # Если передан список ID, фильтруем строки по этому списку
        if isinstance(ids_to_search, list):
            info(f"Фильтрация строк по списку ID: {ids_to_search}")
            filtered_df = df[df['ID дисциплины БУП ППК (АСАВ)'].astype(str).isin([str(id) for id in ids_to_search])]
        else:
            # Если передан один ID, фильтруем только по одному значению
            single_id = str(ids_to_search)
            info(f"Фильтрация строки по ID: {single_id}")
            filtered_df = df[df['ID дисциплины БУП ППК (АСАВ)'].astype(str) == single_id]

        if filtered_df.empty:
            warning(f"Не найдено строк с указанными ID: {ids_to_search}")
        return filtered_df

    else:
        warning(f"Неизвестный режим обработки: {mode}. Будет обработан весь DataFrame.")
        return df
