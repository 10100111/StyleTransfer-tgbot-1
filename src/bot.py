import telebot
from src.config import TOKEN, endpoint, herokuapi
from flask import Flask, request
import os
from src.transfer import process
# import heroku3


def init_and_start_bot():
    bot = telebot.TeleBot(TOKEN, threaded=False)
    server = Flask(__name__)
    users = {}

    @server.route('/' + TOKEN, methods=['POST'])
    def get_message():
        bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
        return "!", 200

    @server.route("/")
    def webhook():
        bot.remove_webhook()
        bot.set_webhook(url=endpoint + TOKEN)
        return "!", 200

    @bot.message_handler(commands=['start'])
    def start_command(message):
        bot.send_message(message.chat.id, 'Hi! im a style transfer bot!')

    @bot.message_handler(commands=['help'])
    def help_command(message):
        bot.send_message(message.chat.id, '/style - at first send content image then style image')

    @bot.message_handler(commands=['style'])
    def style(message):
        users[message.chat.id] = {'started': True, 'content': False, 'style': False}
        bot.send_message(message.chat.id, 'So send me content image')

    @bot.message_handler(content_types=['photo'])
    def process_images(message):
        if message.chat.id in users:
            if not users[message.chat.id]['content']:
                users[message.chat.id]['content'] = message.photo[-1].file_id
                bot.send_message(message.chat.id, 'Send me style image')
            elif not users[message.chat.id]['style']:
                users[message.chat.id]['style'] = message.photo[-1].file_id
                bot.send_message(message.chat.id, 'Transfer started, wait ~1 min')
                call_transfer(message.chat.id)
                with open(str(message.chat.id) + '_styled.jpg', 'rb') as f:
                    bot.send_photo(message.chat.id, f)
                remove_data(message.chat.id)
#                 heroku3.from_key(herokuapi).apps()[0].dynos()[0].restart()

    def remove_data(user):
        for i in ['_content.jpg', '_style.jpg', '_styled.jpg']:
            os.remove(str(user) + i)
        del users[user]

    def call_transfer(user):
        download(user)
        process(str(user) + '_content.jpg', str(user) + '_style.jpg', str(user) + '_styled.jpg')

    def download(user):
        content_id = bot.get_file(users[user]['content'])
        with open(str(user) + '_content.jpg', 'wb') as f:
            f.write(bot.download_file(content_id.file_path))
        style_id = bot.get_file(users[user]['style'])
        with open(str(user) + '_style.jpg', 'wb') as f:
            f.write(bot.download_file(style_id.file_path))

    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
