from utils.config_loader import load_config
from data_processing.pud_loader import load_pud_data
from data_processing.analysis import process_puds
from data_processing.model_loader import load_model

def main():
    # Загрузка конфигурации
    config = load_config("config.yaml")

    # Загрузка данных ПУДов
    df_cleaned = load_pud_data(config)

    # Загрузка модели LLaMA
    model_path = config['model_path']
    model, tokenizer = load_model(model_path)

    # Проверка на успешную загрузку модели и токенизатора
    if model is None or tokenizer is None:
        raise RuntimeError("Не удалось загрузить модель или токенизатор.")

    # Обработка ПУДов и генерация навыков для каждого файла
    process_puds(config, df_cleaned, model, tokenizer)

if __name__ == "__main__":
    main()
