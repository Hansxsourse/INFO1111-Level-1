import numpy as np
import torch
from tools.flappy_bird_game import FlappyBird
from tools.model import Net
import cv2 as cv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set device gpu or cpu

model = Net()
model.load_state_dict(torch.load("model_save/flappy_bird.pt"))  # load trained model
model.to(device)  # move to cuda or cpu
model.eval()  # set model to test model

environment = FlappyBird()  # flappy bird game object

image, _, _ = environment.next_frame(0)  # first frame
image = image[:288, :int(512*0.79)]  # crop the image, cut off the bottom ground space
image = cv.resize(image, (84, 84))  # resize
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # convert to gray
_, image = cv.threshold(image, 1, 255, cv.THRESH_BINARY)

image = image[None, :, :].astype(np.float32)
image = torch.from_numpy(image)  # convert to pytorch tensor
image = image.to(device)  # move image to cuda or gpu

state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]  # make the input size as 4 * height * width

while True:
    preds = model(state)[0]  # inference from model
    action = torch.argmax(preds)  # convert to action value  0 or 1

    next_image, _, _ = environment.next_frame(action)   # place action to the game and get next frame
    next_image = next_image[:288, :int(512 * 0.79)]  # crop the image, cut off the bottom ground space
    next_image = cv.resize(next_image, (84, 84))  # resize
    next_image = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)  # convert to gray\
    _, next_image = cv.threshold(next_image, 1, 255, cv.THRESH_BINARY)
    next_image = next_image[None, :, :].astype(np.float32)
    next_image = torch.from_numpy(next_image)  # convert to pytorch tensor
    next_image = next_image.to(device)  # move to cuda or cpu

    next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]  # same set next state by the frame

    state = next_state  # update