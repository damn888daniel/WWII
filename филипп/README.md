# Филипп - Model Training & Evaluation

## Зона ответственности
- Обучение классификатора
- Оценка метрик модели
- Инференс на новых данных

## Файлы
- `src/train_classifier.py` - обучение логистической регрессии
- `src/predict.py` - предсказание на новых изображениях
- `models/clip_fake_detector.pkl` - обученная модель
- `models/metrics.json` - метрики качества

## Задачи
- [ ] Обучить логистическую регрессию на CLIP эмбеддингах
- [ ] Оценить метрики (accuracy, precision, recall, F1)
- [ ] Построить confusion matrix
- [ ] Реализовать инференс на новых данных
- [ ] Провести error analysis
