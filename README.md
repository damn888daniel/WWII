# Детектор AI-сгенерированных изображений WWII

Курсовой проект по детекции синтетических изображений Второй мировой войны с использованием мультимодальных эмбеддингов CLIP.

## Описание проекта

Проект решает задачу бинарной классификации изображений на **реальные исторические фотографии WWII** и **синтетические (AI-сгенерированные)** изображения.

### Подход

1. **Данные**
   - **Реальные изображения**: ~800 исторических фотографий WWII из Wikimedia Commons (NARA архивы)
   - **Синтетические изображения**: 200 изображений, сгенерированных с помощью SDXL-Turbo с промптами на тему альтернативной истории WWII

2. **Модель**
   - Извлечение мультимодальных эмбеддингов с помощью CLIP (openai/clip-vit-large-patch14)
   - Объединение image и text embeddings в единый вектор признаков
   - Классификация с помощью логистической регрессии

3. **Пайплайн**
   - Генерация синтетических данных (SDXL-Turbo)
   - Сборка объединенного манифеста с разделением на train/val/test
   - Извлечение CLIP эмбеддингов
   - Обучение классификатора
   - Инференс на новых изображениях

## Структура проекта

```
.
├── data/
│   ├── real/
│   │   ├── raw/              # ~800 реальных изображений WWII
│   │   └── meta/
│   │       └── real_manifest.jsonl
│   ├── synth/
│   │   ├── raw/              # 200 синтетических изображений
│   │   └── meta/
│   │       └── synth_manifest.jsonl
│   ├── manifest.csv          # Объединенный датасет с train/val/test split
│   ├── embeddings.npz        # CLIP эмбеддинги
│   ├── generate_synthetic_sdxl.py
│   └── download_real_wikimedia.py
├── src/
│   ├── build_manifest.py     # Объединение манифестов и сплит
│   ├── extract_embeddings.py # Извлечение CLIP эмбеддингов
│   ├── train_classifier.py  # Обучение классификатора
│   └── predict.py            # Инференс
├── models/
│   ├── clip_fake_detector.pkl  # Обученная модель
│   └── metrics.json            # Метрики (accuracy, precision, recall, F1)
├── webapp/
│   ├── app.py                  # Flask backend
│   ├── templates/
│   │   └── index.html          # HTML шаблон
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css       # Винтажные стили
│   │   └── js/
│   │       └── main.js         # Frontend логика
│   └── uploads/                # Временные загрузки
├── requirements.txt
└── README.md
```

## Установка

1. Создать виртуальное окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### 1. Генерация синтетических данных

```bash
python data/generate_synthetic_sdxl.py --num-images 200 --steps 4 --guidance 0.0
```

Параметры:
- `--num-images`: количество изображений для генерации
- `--steps`: количество шагов диффузии (4 для SDXL-Turbo)
- `--guidance`: CFG scale (0.0 для Turbo)
- `--model-id`: модель диффузии (по умолчанию stabilityai/sdxl-turbo)

### 2. Сборка датасета

```bash
python src/build_manifest.py --balance
```

Создает `data/manifest.csv` с колонками: `image_path`, `caption`, `label`, `split`

Параметры:
- `--balance`: балансировка классов (обрезка до размера меньшего класса)
- `--val-size`: доля валидации (по умолчанию 0.1)
- `--test-size`: доля теста (по умолчанию 0.1)

### 3. Извлечение эмбеддингов CLIP

```bash
python src/extract_embeddings.py
```

Извлекает мультимодальные эмбеддинги (image + text) для всех изображений и сохраняет в `data/embeddings.npz`

Параметры:
- `--model`: модель CLIP (по умолчанию openai/clip-vit-large-patch14)
- `--batch-size`: размер батча (по умолчанию 8)

### 4. Обучение классификатора

```bash
python src/train_classifier.py
```

Обучает логистическую регрессию на CLIP эмбеддингах и сохраняет модель в `models/clip_fake_detector.pkl`

Метрики сохраняются в `models/metrics.json`

### 5. Инференс

```bash
python src/predict.py \
  --image path/to/image.jpg \
  --caption "описание изображения"
```

Пример:
```bash
python src/predict.py \
  --image data/real/raw/100396877.jpg \
  --caption "WWII soldiers in Europe, 1944"
```

Вывод:
```json
{
  "prediction": "real",
  "prob_fake": 0.05,
  "prob_real": 0.95
}
```

## Метрики модели

После обучения метрики доступны в `models/metrics.json`:

```json
{
  "train": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "val": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "test": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

**Результаты:**
- Train: 320 примеров (160 real, 160 synthetic), 100% accuracy
- Val: 40 примеров (20 real, 20 synthetic), 100% accuracy
- Test: 40 примеров (20 real, 20 synthetic), 100% accuracy

Confusion Matrix (Test):
```
              precision    recall  f1-score   support
        real       1.00      1.00      1.00        20
   synthetic       1.00      1.00      1.00        20
```

## Зависимости

Основные библиотеки:
- `torch` - PyTorch для нейросетей
- `transformers` - CLIP модель от Hugging Face
- `diffusers` - SDXL-Turbo для генерации синтетики
- `scikit-learn` - логистическая регрессия и метрики
- `pandas` - работа с данными
- `pillow` - обработка изображений

Полный список в `requirements.txt`

## Результаты

Проект демонстрирует, что мультимодальные эмбеддинги CLIP в сочетании с простым линейным классификатором эффективно различают реальные исторические фотографии WWII и AI-сгенерированные изображения на тему альтернативной истории.

Ключевые преимущества подхода:
- Использование как визуальной, так и текстовой информации
- Простота и интерпретируемость модели
- Хорошая обобщающая способность на тестовой выборке

## Веб-приложение

Проект включает веб-интерфейс в стиле военных документов 1940-х годов для удобного тестирования модели.

### Запуск веб-приложения

```bash
cd webapp
python app.py
```

Приложение будет доступно по адресу: **http://127.0.0.1:5001**

### Возможности веб-приложения

- Загрузка изображений через drag & drop или выбор файла
- Ввод описания изображения (опционально)
- Анализ в реальном времени с использованием CLIP + логистическая регрессия
- Визуализация результатов:
  - Вердикт: AUTHENTIC / SYNTHETIC
  - Уверенность модели в процентах
  - Вероятности для каждого класса
  - Детальная интерпретация результата
- Винтажный дизайн в стиле секретных военных документов

### Структура веб-приложения

```
webapp/
├── app.py                 # Flask backend
├── templates/
│   └── index.html         # HTML шаблон
├── static/
│   ├── css/
│   │   └── style.css      # Винтажные стили
│   └── js/
│       └── main.js        # Frontend логика
└── uploads/               # Временные загрузки (автоматически создается)
```

## Автор

Курсовая работа по мультимодальным моделям
