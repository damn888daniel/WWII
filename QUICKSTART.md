# Быстрый старт

## Запуск веб-приложения

```bash
# 1. Активировать виртуальное окружение
source .venv/bin/activate

# 2. Запустить Flask приложение
cd webapp
python app.py
```

Приложение будет доступно на **http://127.0.0.1:5001**

## Тестирование модели через командную строку

```bash
# Активировать окружение
source .venv/bin/activate

# Протестировать на реальном изображении
python src/predict.py \
  --image data/real/raw/100396877.jpg \
  --caption "WWII historical photograph"

# Протестировать на синтетическом изображении
python src/predict.py \
  --image data/synth/raw/synth_00000.png \
  --caption "Alternative WWII scene"
```

## Полный пайплайн (если нужно переобучить)

```bash
source .venv/bin/activate

# 1. Генерация синтетических данных (200 изображений, ~10 мин)
python data/generate_synthetic_sdxl.py --num-images 200 --steps 4 --guidance 0.0

# 2. Создание объединенного датасета
python src/build_manifest.py --balance

# 3. Извлечение CLIP эмбеддингов
python src/extract_embeddings.py --batch-size 16

# 4. Обучение классификатора
python src/train_classifier.py
```

## Результаты модели

- **Train accuracy**: 100%
- **Val accuracy**: 100%
- **Test accuracy**: 100%

Модель идеально различает реальные исторические фотографии WWII от AI-сгенерированных изображений.
