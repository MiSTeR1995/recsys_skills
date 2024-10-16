import os
import torch
from tqdm import tqdm
import pandas as pd
from data_processing.pud_loader import filter_rows_by_mode
from data_processing.rec_generation import generate_recommendations
from utils.logger import info, success, warning, error, highlight, bright


def get_processed_ids(output_folder, enable_id_check=True):
    """
    Получает уже обработанные ID из CSV файлов в папке с результатами.

    :param output_folder: Папка для сохранения результатов.
    :param enable_id_check: Если True, выполняется проверка уже обработанных ID.
    :return: Множество с обработанными ID.
    """
    processed_ids = set()

    if not enable_id_check:
        info("Проверка ID отключена. Все строки будут обработаны.")
        return processed_ids  # Возвращаем пустое множество, если проверка отключена

    # Проверяем все файлы в папке результатов
    for file in os.listdir(output_folder):
        output_file = os.path.join(output_folder, file)
        if os.path.isfile(output_file):
            try:
                df_output = pd.read_csv(output_file, sep=";", usecols=[0])  # Предполагается, что ID — первый столбец
                processed_ids.update(df_output.iloc[:, 0].astype(str).tolist())
                count = len(df_output)
                info(f"Найдено совпадение по ID {count} ранее обработанных строк в файле {output_file}.")
            except Exception as e:
                warning(f"Не удалось прочитать файл {output_file} для загрузки обработанных ID: {e}")

    return processed_ids


def process_puds(config, df_cleaned, model, tokenizer):
    try:
        # Указываем папку для сохранения результатов
        output_folder = config.get("output_folder", "results")
        os.makedirs(output_folder, exist_ok=True)

        # Получаем уже обработанные ID
        enable_id_check = config.get('processing', {}).get('enable_id_check', False)
        processed_ids = get_processed_ids(output_folder, enable_id_check)

        # Применяем фильтрацию строк по режиму
        df_filtered = filter_rows_by_mode(df_cleaned, config)

        total_rows = len(df_filtered)
        with tqdm(total=total_rows, desc="Общий прогресс обработки", unit="строка") as pbar:
            for index, row in df_filtered.iterrows():
                try:
                    # Получаем ID ПУДа
                    pud_id = str(row["ID дисциплины БУП ППК (АСАВ)"]).strip()

                    # Пропускаем уже обработанные строки
                    if enable_id_check and pud_id in processed_ids:
                        continue

                    # Обрабатываем строку
                    process_row(index, row, processed_ids, model, tokenizer, output_folder, config)

                    pbar.update(1)
                except Exception as e:
                    error(f"Ошибка при обработке строки {index}: {e}")

    except Exception as e:
        error(f"Ошибка при инициализации процесса обработки ПУДов: {e}")


def process_row(index, row, processed_ids, model, tokenizer, output_folder, config):
    try:
        # Используем "ID дисциплины БУП ППК (АСАВ)" как уникальный идентификатор для дисциплины
        pud_id = str(row["ID дисциплины БУП ППК (АСАВ)"]).strip()

        # Извлекаем ключевые столбцы
        smartplan_id = str(row.get("ID дисциплины БУП ППК (SmartPlan)", "")).strip()
        discipline_name = str(row.get("Русскоязычное название дисциплины", "")).strip()
        annotation = str(row.get("Аннотация", "")).strip()
        sections = str(row.get("Список разделов (названия и описания)", "")).strip()
        learning_outcomes = str(row.get("Список планируемых результатов обучения РПУДа", "")).strip()

        # Проверка, есть ли хоть какая-то информация для обработки
        if not annotation and not sections and not learning_outcomes:
            warning(f"Отсутствует информация для ПУДа (ID: {pud_id}). Пропуск строки.")
            return  # Пропускаем строку, если все данные отсутствуют

        # Формируем полный текст для генерации навыков, включая название дисциплины
        pud_description = f"Название дисциплины: {discipline_name}. "
        if annotation:
            pud_description += f"Аннотация: {annotation}. "
        if sections:
            pud_description += f"Разделы курса: {sections}. "
        if learning_outcomes:
            pud_description += f"Результаты обучения: {learning_outcomes}."

        bright(f"Описание ПУДа (ID: {pud_id}) для строки {index}: {pud_description}")

        # Генерация навыков с использованием LLaMA
        skills = generate_recommendations(pud_description, model, tokenizer, config)
        highlight(f"Сгенерированные навыки для строки {index} (ID: {pud_id}): {skills}")

        # Создание нового DataFrame для сохранения нужных столбцов
        row_to_save = {
            "ID дисциплины БУП ППК (АСАВ)": pud_id,
            "ID дисциплины БУП ППК (SmartPlan)": smartplan_id,
            "Русскоязычное название дисциплины": discipline_name,
            "Аннотация": annotation,
            "Список разделов (названия и описания)": sections,
            "Список планируемых результатов обучения РПУДа": learning_outcomes,
            "LLM_Skills": skills
        }

        # Преобразуем строку в DataFrame для сохранения
        row_df = pd.DataFrame([row_to_save])

        # Сохраняем строку в CSV файл
        output_file = os.path.join(output_folder, "processed_puds.csv")
        row_df.to_csv(output_file, mode='a', header=not os.path.isfile(output_file), index=False, encoding="utf-8-sig", sep=";")
        processed_ids.add(pud_id)
        info(f"Строка {index} (ID: {pud_id}) сохранена в файл {output_file}.")

    except Exception as e:
        error(f"Ошибка при обработке строки {index} (ID: {pud_id}): {e}")
