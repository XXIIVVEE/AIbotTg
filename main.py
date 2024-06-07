import telebot
import cv2
import numpy as np
import tensorflow as tf
import os

# Вставьте ваш токен, полученный от @BotFather
TOKEN = '7327632875:AAFV3_wPx1qEa7YcF-TXpuiL0fmE68uCgxM'
bot = telebot.TeleBot(TOKEN)

# Загрузка предобученной модели SSD MobileNet
MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_320x320/saved_model'
if not os.path.exists(MODEL_DIR):
    raise IOError(f"Model directory {MODEL_DIR} does not exist.")
model = tf.saved_model.load(MODEL_DIR)

# Загрузка меток классов объектов
LABELS_PATH = 'mscoco_label_map.pbtxt'
if not os.path.exists(LABELS_PATH):
    raise IOError(f"Labels file {LABELS_PATH} does not exist.")
with open(LABELS_PATH) as f:
    labels = {int(line.split(': ')[1]): next(f).split(': ')[1].strip() for line in f if 'id' in line}

# Функция для анализа изображения и подсчета объектов
def analyze_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    object_count = {}
    for idx, score in enumerate(detection_scores):
        if score > 0.5:  # Считаем только объекты с уверенностью больше 50%
            label = labels[detection_classes[idx]]
            if label in object_count:
                object_count[label] += 1
            else:
                object_count[label] = 1

    return object_count

# Обработчик сообщений с фото
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Загружаем фото
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Сохраняем фото локально
    image_path = 'image.jpg'
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    
    # Анализируем фото
    object_count = analyze_image(image_path)

    # Формируем ответ
    response = "Объекты на изображении:\n"
    for obj, count in object_count.items():
        response += f"{obj}: {count}\n"

    # Отправляем результат пользователю
    bot.reply_to(message, response)

# Запуск бота
bot.polling()
