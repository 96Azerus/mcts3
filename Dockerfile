# Dockerfile v1.1
# Используем официальный образ Python
FROM python:3.11

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей сначала, чтобы использовать кэш Docker
COPY requirements.txt requirements.txt

# Устанавливаем зависимости
# Обновляем pip и устанавливаем зависимости без кэша
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Создаем пользователя без пароля и лишних вопросов
# Используем /sbin/nologin для большей безопасности
RUN adduser --system --no-create-home --group --disabled-password --gecos "" appuser

# Копируем все файлы приложения (включая src)
COPY . .

# Устанавливаем владельца для всего /app
# Делаем это после копирования всех файлов
RUN chown -R appuser:appuser /app

# Устанавливаем PYTHONPATH, чтобы Python находил модули в /app (где лежит src)
ENV PYTHONPATH=/app

# Переключаемся на непривилегированного пользователя
USER appuser

# Указываем Flask, где искать приложение (app.py в корне /app)
ENV FLASK_APP=app.py
# Указываем порт по умолчанию (будет переопределен через env var PORT)
# ENV PORT=8080
# Устанавливаем режим работы Gunicorn (production рекомендуется, но для Codespace можно development)
# ENV APP_ENV=production

# Команда для запуска приложения с использованием Gunicorn
# Gunicorn будет запускаться от имени appuser
# Используем exec для того, чтобы Gunicorn стал PID 1 в контейнере
# Используем переменные окружения PORT и WEB_CONCURRENCY
# Добавляем --timeout для долгих MCTS ходов
CMD exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers ${WEB_CONCURRENCY} --timeout 120 --log-level info app:app
