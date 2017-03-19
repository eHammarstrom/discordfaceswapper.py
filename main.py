import discord
import asyncio
import os
import cv2
import numpy as np
import re
import math

from PIL import Image
import requests
from io import BytesIO

client = discord.Client()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
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

        if 'load' in message.content.strip(urls[0]):
            await face_load_handler(message, urls[0])
        elif 'print' in message.content.strip(urls[0]):
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

    faces = retrieve_faces(BytesIO(response.content))

    if len(faces) < 1:
        x = int(img_final.width / 2) - int(img_face_replace.width / 2)
        y = img_final.height - int(img_face_replace.height / 2)

        await client.send_file(message.channel, 
                image_to_mem_buf(image_place(
                    img_final,
                    img_face_replace,
                    (x, y, img_face_replace.width, img_face_replace.height),
                    0), 'png'))

        return

    for (x, y, w, h) in faces:
        eyes = retrieve_eyes_on_face(
                BytesIO(response.content), x, y, w, h)

        if len(eyes) == 2:
            # computes tilt of eyes to guestimate face tilt
            deltaX = eyes[0][0] - eyes[1][0]
            deltaY = eyes[1][1] - eyes[0][1] # swap index for image y coords
            deg = math.degrees(math.atan(deltaY / deltaX))
        elif len(eyes) == 3:
            await client.send_message(message.channel, 'Illuminati confirmed.')
        else:
            deg = 180 # for giggles

        height = h * (img_face_replace.height/img_face_replace.width)

        img_final = image_place(img_final,
                img_face_replace,
                (x, y, w, int(height)),
                deg)

    await client.send_file(message.channel, image_to_mem_buf(img_final, 'png'))

#
# Superimpose image on image given position and rotation
#
def image_place(main_image, place_image, xywh_tup, deg):
    place_image = place_image.resize((xywh_tup[2], xywh_tup[3]))
    place_image = place_image.rotate(deg, resample=Image.BICUBIC)
    main_image.paste(place_image, (xywh_tup[0], xywh_tup[1]), mask=place_image)

    return main_image

#
# Retrieves all coordinates of faces on given image
#
def retrieve_faces(image_fp):
    cv_mat_gray = mem_buf_to_cv2_mat(image_fp)

    # object detection using face cascade on cv matrix
    faces = face_cascade.detectMultiScale(cv_mat_gray, 1.1, 5)

    print("faces", faces)

    return faces

#
# Parameter: a face (x, y, w, h)
# Output: vector of eyes
#
def retrieve_eyes_on_face(image_fp, x, y, w, h):
    cv_mat_gray = mem_buf_to_cv2_mat(image_fp)
    cv_mat_face = cv_mat_gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(cv_mat_face)

    print("eyes", eyes)

    return eyes

#
# Transforms Image in buffer to cv2 matrix
#
def mem_buf_to_cv2_mat(image_fp):
    # image file as byte array
    image_buf = np.asarray(bytearray(image_fp.read()), dtype=np.uint8)

    # image buffer to cv matrix
    cv_mat_gray = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)

    return cv_mat_gray

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
