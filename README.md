# StyleTransfer-tgbot

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/timbyxty/StyleTransfer-tgbot)

# Модель (реализация [статьи](https://arxiv.org/pdf/1703.06953.pdf))

Обучена стилизовать картинки под уже стилизованные с помощью модели Гатиса(предобученная VGG16){100 итераций на картинку}, на датасете [COCO](https://cocodataset.org/){H:256, W:256} 1 полную эпоху на 10 картинках стиля.

   1   |   2   |   3   |   4   |   5
:-----:|:-----:|:-----:|:-----:|:-----:|
<img src=https://sun1.ufanet.userapi.com/dGdXLH95xCmt6ut-dB9nH3h3QsQmuLXR0uutzQ/0oJc1awmbqc.jpg width="700">  |<img src=https://sun1.ufanet.userapi.com/AA0Evxr9tBKBwrpxpMNgB4w57gXuSkOH7cmjhA/Y5EqvhbhOHU.jpg width="700">  | <img src=https://sun3.ufanet.userapi.com/8s5FQYLcJbSgplZdTR66iZIByvvqn-TcW7I9PQ/-YvQ5YLjP5Q.jpg width="700"> | <img src=https://sun3.ufanet.userapi.com/KXSjBXRRBezKqOanrKn7TTKI7okwBzSxlY9U-w/ywzPEVIvgdI.jpg width="700"> | <img src=https://sun2.ufanet.userapi.com/TknzZme8L-lQQNT9dQYrzysBieqUOE6HdhHRDg/1hwdGwEhDyk.jpg width="700">

   6   |   7   |   8   |   9   |   10
:-----:|:-----:|:-----:|:-----:|:-----:|
<img src=https://sun3.ufanet.userapi.com/IciG2UGQsKPaqHvqzDX9tTIKIMbgDdw9JWkb1w/FmMXT9J9Kko.jpg width="700">  |<img src=https://sun3.ufanet.userapi.com/OCO4-thZwdcpnNkkp72qvr9CtCs48x7lxktBKQ/hoWP7QPBav4.jpg width="700">  | <img src=https://sun1.ufanet.userapi.com/jXqGuoLwWdTUjnazn9Z1-QvpfupZVpnkRyw-qA/VvASjJDW0pg.jpg width="700"> | <img src=https://sun9-71.userapi.com/vfBhzMWnGV8Dq33JJ2OYYRs2o4vHdSAsv4WWuQ/0dgSkmrwn_o.jpg width="700"> | <img src=https://sun3.ufanet.userapi.com/LbSxakoz50BgLq5goXMhc6CYMG1dk4SKYHjspw/VEHm8m0Hwdk.jpg width="700">


*Код обучения вырезан, оставлен только код для деплоя.*



# Telegram frontend <a href="https://t.me/ebanyivolshebnikBot"><img src=https://sun9-38.userapi.com/c858528/v858528388/1c0f17/l8lwLWnQHg8.jpg width="32"></a>
Связка с телеграмом реализована при помощи библиотеки [pyTelegramBotAPI](https://pypi.org/project/pyTelegramBotAPI/0.3.0/)

Работающий бот доступен в телеграме по никнейму [@ebanyivolshebnikBot](https://t.me/ebanyivolshebnikBot)
