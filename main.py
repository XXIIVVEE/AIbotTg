import io
import telebot
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Токен вашего бота
bot_token = '7327632875:AAFV3_wPx1qEa7YcF-TXpuiL0fmE68uCgxM'
bot = telebot.TeleBot(bot_token)

# Загрузка модели MobileNetV2
model = MobileNetV2(weights='imagenet')

def prepare_image(image: Image.Image):
    # Преобразование изображения для модели
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = image_array.reshape((1, *image_array.shape))
    return image_array

def predict_image(image_array):
    # Получение предсказаний
    predictions = model.predict(image_array)
    predictions = decode_predictions(predictions, top=3)[0]
    return predictions

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Пожалуйста, отправьте мне картинку для обработки.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Отправка сообщения о начале обработки
    processing_message = bot.reply_to(message, "Подождите, картинка обрабатывается...")
    try:
        # Получение изображения от пользователя
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Открытие и обработка изображения
        image_stream = io.BytesIO(downloaded_file)
        image = Image.open(image_stream)
        image_array = prepare_image(image)

        # Предсказание и фильтрация результатов
        predictions = predict_image(image_array)
        threshold = 0.1
        filtered_predictions = [(class_id, title, score) for class_id, title, score in predictions if score >= threshold]

        # Выбор и отправка предсказания с самой высокой вероятностью
        if filtered_predictions:
            highest_prediction = max(filtered_predictions, key=lambda item: item[2])
            class_id, title, score = highest_prediction
            response = f"Название: {title}, Вероятность: {int(score * 100)}%"
            # Редактирование сообщения с результатами
            bot.edit_message_text(chat_id=processing_message.chat.id, message_id=processing_message.message_id, text=response)
        else:
            # Редактирование сообщения об отсутствии результатов
            bot.edit_message_text(chat_id=processing_message.chat.id, message_id=processing_message.message_id, text="Нет предсказаний с вероятностью выше установленного порога.")
    except Exception as e:
        # Редактирование сообщения об ошибке
        bot.edit_message_text(chat_id=processing_message.chat.id, message_id=processing_message.message_id, text=f"Произошла ошибка: {e}")

# Запуск бота
bot.polling()
