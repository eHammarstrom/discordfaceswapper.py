import discord
import asyncio
import os
import cv2
import numpy as np
import re

from PIL import Image
import requests
from io import BytesIO

client = discord.Client()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img_face_replace = Image.open('face.png')

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

@client.event
async def on_message(message):
    if message.content.startswith('!face'):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content)

        if len(message.attachments) > 0:
            urls = [message.attachments[0]['url']]

        if 'load' in message.content:
            await face_load_handler(message, urls[0])
        elif 'print' in message.content:
            await client.send_file(message.channel,
                    image_to_mem_buf(img_face_replace, 'png'))
        else:
            await face_replace_handler(message, urls[0])

#
# Loads image from url and replaces img_face_replace with given image
#
async def face_load_handler(message, url):
    response = requests.get(url)

    if response.status_code == 200 and \
            'image' in response.headers['content-type'].lower():
        global img_face_replace
        img_face_replace = Image.open(BytesIO(response.content))

        await client.send_message(message.channel, 'Image was successfully loaded.')
    else:
        await client.send_message(message.channel, 'There was an error loading the image.')

#
# Loads image from url and replaces all faces with img_face_replace
#
async def face_replace_handler(message, url):
    response = requests.get(url)
    img_file = BytesIO(response.content)
    img_final = Image.open(img_file)

    for (x, y, w, h) in retrieve_faces(BytesIO(response.content)):
        temp = img_face_replace
        height = h * (img_face_replace.height/img_face_replace.width)
        temp = temp.resize((w, int(height)))
        img_final.paste(temp, (x, y), mask=temp)

    await client.send_file(message.channel, image_to_mem_buf(img_final, 'png'))

#
# Retrieves all coordinates of faces on given image
#
def retrieve_faces(image_fp):
    # image file as byte array
    image_buf = np.asarray(bytearray(image_fp.read()), dtype=np.uint8)

    # image buffer to cv matrix
    cv_mat_gray = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)

    # object detection using face cascade on cv matrix
    faces = face_cascade.detectMultiScale(cv_mat_gray, 1.1, 5)

    return faces

#
# Transforms pillow Image to in-memory buffer image
#
def image_to_mem_buf(image, ftype):
    buf = BytesIO()
    buf.seek(0)
    image.save(buf, ftype)
    buf.name = '.' + ftype
    buf.seek(0)
    return buf

client.run(os.environ['DISCORD_TOKEN'])
