# Гайд по коммитам для команды

## Как правильно коммитить свою работу

### 1️⃣ Работай в своей папке
Каждый участник работает в своей директории:
- **Даня** → папка `даня/`
- **Боря** → папка `боря/`
- **Филипп** → папка `филипп/`
- **Илья** → папка `илья/`

### 2️⃣ Примеры коммитов

#### Для Дани (работа с данными):
```bash
cd /path/to/WWII
git add даня/
git commit -m "Collect 800 real WWII images from Wikimedia Commons"
git push
```

Другие примеры:
```bash
git commit -m "Generate 200 synthetic WWII images with SDXL-Turbo"
git commit -m "Create manifest files for real and synthetic datasets"
git commit -m "Add data quality checks and validation scripts"
```

#### Для Бори (обработка данных):
```bash
cd /path/to/WWII
git add боря/
git commit -m "Build unified manifest with train/val/test split"
git push
```

Другие примеры:
```bash
git commit -m "Extract CLIP embeddings for all images"
git commit -m "Implement data preprocessing pipeline"
git commit -m "Add embedding dimension validation"
```

#### Для Филиппа (ML модель):
```bash
cd /path/to/WWII
git add филипп/
git commit -m "Train logistic regression on CLIP embeddings"
git push
```

Другие примеры:
```bash
git commit -m "Evaluate model metrics: 100% accuracy on test set"
git commit -m "Implement inference script for new images"
git commit -m "Add confusion matrix visualization"
```

#### Для Ильи (веб-приложение):
```bash
cd /path/to/WWII
git add илья/
git commit -m "Create Flask API for model inference"
git push
```

Другие примеры:
```bash
git commit -m "Implement drag & drop file upload interface"
git commit -m "Add vintage 1940s style CSS to webapp"
git commit -m "Create result visualization with confidence scores"
```

### 3️⃣ Когда нужно изменить общие файлы

Если нужно изменить файлы вне личных папок:

```bash
# Изменяем requirements.txt (новые зависимости)
git add requirements.txt
git commit -m "Add Flask and Werkzeug to requirements"

# Изменяем README.md (документация)
git add README.md
git commit -m "Update README with team structure"

# Изменяем общие скрипты в src/
git add src/
git commit -m "Fix bug in extract_embeddings.py"
```

### 4️⃣ Проверка перед коммитом

```bash
# Посмотреть, что изменилось
git status

# Посмотреть конкретные изменения
git diff

# Посмотреть историю
git log --oneline -5
```

### 5️⃣ Полный workflow

```bash
# 1. Убедиться, что у тебя последняя версия
git pull

# 2. Сделать свою работу в своей папке
cd даня/  # или боря/, филипп/, илья/
# ... работаем ...

# 3. Посмотреть изменения
git status
git diff

# 4. Добавить изменения
git add даня/  # или свою папку

# 5. Создать коммит с понятным сообщением
git commit -m "Описание того, что сделано"

# 6. Отправить на GitHub
git push
```

### 6️⃣ Важно!

- ✅ Пиши понятные commit messages (что сделано, а не как)
- ✅ Коммить часто (каждая логическая часть работы = отдельный коммит)
- ✅ Коммитить только свою папку (если не договорились иначе)
- ❌ Не коммить __pycache__, .pyc файлы (они в .gitignore)
- ❌ Не коммить большие файлы без Git LFS

### 7️⃣ Если что-то пошло не так

```bash
# Отменить последний коммит (но сохранить изменения)
git reset --soft HEAD~1

# Отменить изменения в файле
git checkout -- filename

# Посмотреть, что будет закоммичено
git diff --staged
```

## Вопросы?
Спрашивай в общем чате!
