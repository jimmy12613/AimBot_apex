#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
import mss.tools
import math
import numpy as np
import cv2
import mouse
import keyboard
import time


# In[38]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[39]:


def getcoords(enemy):
    try:
        return [float(results.xyxy[0][enemy, 0]),
                float(results.xyxy[0][enemy, 1]),
                float(results.xyxy[0][enemy, 2]),
                float(results.xyxy[0][enemy, 3])]
    except IndexError:
        return None


def move_mousey(y):
    mouse.move(0, y, absolute=False, duration=0)

    #windll.user32.mouse_event(
        #c_uint(0x0001),
        #c_uint(0),
        #c_uint(y),
        #c_uint(0),
        #c_uint(0)
    #)


def move_mousex(x):
    mouse.move(x, 0, absolute=False, duration=0)

    #windll.user32.mouse_event(
        #c_uint(0x0001),
        #c_uint(x),
        #c_uint(0),
        #c_uint(0),
        #c_uint(0)
    #)


# In[40]:


with mss.mss() as sct:
    # Use the first monitor, change to desired monitor number
    dimensions = sct.monitors[1]
    SQUARE_SIZE = 800

    # Part of the screen to capture
    monitor = {"top": int((dimensions['height'] / 2) - (SQUARE_SIZE / 2)),
               "left": int((dimensions['width'] / 2) - (SQUARE_SIZE / 2)),
               "width": SQUARE_SIZE,
               "height": SQUARE_SIZE}

    while True:

        start_time = time.time()  # start time of the loop

        # Screenshot
        sct_img = sct.grab(monitor)

        # Convert to
        img = np.array(sct_img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Inference
        results = model(img)
        model.conf = 0.6

    # Reading and determining the closest enemy to crosshair

        # Get number of enemies / length of the .xyxy[0] array
        enemyNum = results.xyxy[0].shape[0]

        # Reset distances array to prevent repeating items
        distances = []

        # Cycle through enemies and draw lines from center to center of their head
        for i in range(enemyNum):
            enemy = getcoords(i)

            x1 = enemy[0]
            x2 = enemy[2]
            y1 = enemy[1]
            y2 = enemy[3]

            centerX = (x2 - x1) / 2 + x1
            centerY = (y2 - y1) / 2 + y1

            distance = math.sqrt(((centerX - 300) ** 2) + ((centerY - 300) ** 2))
            distances.append(distance)

            cv2.line(results.imgs[0], (int(centerX), int(centerY)), (300, 300), (255, 0, 0), 1, cv2.LINE_AA)

        # Get the shortest distance from the array (distances)
        closest = 1000



        # Get the smallest value of the array
        for i in range(0, len(distances)):
            if distances[i] < closest:
                closest = distances[i]
                closestEnemy = i

        # Getting the coordinates of the closest enemy
        if len(distances) != 0:
            enemyCoords = getcoords(closestEnemy)

            x1 = enemyCoords[0]
            x2 = enemyCoords[2]
            y1 = enemyCoords[1]
            y2 = enemyCoords[3]

            Xenemycoord = (x2 - x1) / 2 + x1
            Yenemycoord = (y2 - y1) / 2 + y1

    # Moving mouse to enemy

            difx = Xenemycoord - (SQUARE_SIZE / 2)
            dify = Yenemycoord - (SQUARE_SIZE / 2)

        if keyboard.is_pressed('/'):
            if difx < 0:
                move_mousex(int(difx * 2))

            elif difx > 0:
                move_mousex(int(abs(difx) * 2))

            if dify < 0:
                move_mousey(int(dify * 2))

            elif dify > 0:
                move_mousey(int(abs(dify) * 2))

        # Display the picture
        results.display(render=True)
        cv2.imshow('', results.imgs[0])

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sct.close()
            break

        print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


# In[40]:




