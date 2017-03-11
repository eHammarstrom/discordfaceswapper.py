import discord
import asyncio
import os
import cv2
import numpy as np

from PIL import Image
import requests
from io import BytesIO

client = discord.Client()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

@client.event
async def on_message(message):
    if message.content.startswith('!face'):
        if len(message.attachments) > 0:
            await face_handler(message, message.attachments[0]['url'])
        else:
            await face_handler(message, message.content.strip('!face').strip('\n').strip(' '))

async def face_handler(message, url):
    face_replacement = Image.open('face.png')

    response = requests.get(url)
    file_obj = BytesIO(response.content)
    file_as_buf = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)

    file_mod = BytesIO(response.content)
    image = Image.open(file_mod)

    cv_mat_gray = cv2.imdecode(file_as_buf, cv2.IMREAD_GRAYSCALE)

    faces = face_cascade.detectMultiScale(cv_mat_gray, 1.1, 5)

    for (x, y, w, h) in faces:
        temp = face_replacement
        height = h * (face_replacement.height/ face_replacement.width)
        temp = temp.resize((w, int(height)))
        image.paste(temp, (x, y), mask=temp)

    file_mod.seek(0)
    image.save(file_mod, 'jpeg')
    file_mod.name = 'test.jpg'
    file_mod.seek(0)

    await client.send_file(message.channel, file_mod)

client.run(os.environ['DISCORD_TOKEN'])
