import argparse
import os
import random

import numpy as np
import torch
# from torch.optim import optimizer
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch import optim
import cv2 as cv
from PIL import Image

from tools.model import Net

from tools.flappy_bird_game import FlappyBird
from tqdm import trange


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channel", type=int, default=4, help="num of channel of input imgs")
    parser.add_argument("--num_epoch", type=int, default=2000000, help="num of channel of input imgs")
    parser.add_argument("--num_class", type=int, default=2, help="num of action type for player")
    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--img_size", type=int, default=84, help="image size for training")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon value")
    parser.add_argument("--replay_memory_size", type=int, default=10000, help="replay memory maximum size value")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma for deduction")

    args = parser.parse_args()
    return args


def seed_everything(seed):
    """
    This function we used to fix every part's random seed, to make sure the results are all the same for each run.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, criterion, optimizer_ft, num_epochs):
    arg = args()  # get command line parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set device gpu or cpu

    environment = FlappyBird()
    image, reward, terminal = environment.next_frame(
        0)  # this method will return a frame of image and the reward value, also a bool to indicate if the bird hit the ground or the pipe
    # the parameter for the next_frame method is the action, there are two values to choose: 0 and 1, 0 is not fly while 1 is fly once.

    model.to(device)  # move model to gpu or cpu

    image = cv.cvtColor(cv.flip(np.rot90(image), 0),
                        cv.COLOR_BGR2GRAY)  # convert the image output from pygame frame into proper form
    transform = transforms.Compose([
        transforms.Resize((arg.img_size)),
        transforms.ToTensor(),
    ])  # resize image and convert to tensor.
    # scripted_transforms = torch.jit.script(transforms)
    image = cv.resize(image[:288, :int(512*0.79)], (arg.img_size, arg.img_size))
    image = transform(Image.fromarray(np.uint8(image)))  # image now is a transformed tensor
    image = image.to(device)  # move the image into cuda or cpu
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    epsilon_decrements = np.linspace(0.1, 1e-4, arg.num_epoch)
    Loss_list = []
    replay_memory = []  # to save a series of action
    for epoch in trange(num_epochs):
        pred = model(state)[0]  # get the action predict from model

        # here we use epsilon to avoid the model goes failed to train.
        random_action = random.random() <= epsilon_decrements[epoch]
        if random_action:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(pred)  # convert to 0 or 1

        next_image, next_reward, next_terminal = environment.next_frame(
            action)  # step into next frame using the action predict from model.
        # transform the next image into proper tensor and unpack it to a state
        next_image = cv.cvtColor(cv.flip(np.rot90(next_image), 0),
                                 cv.COLOR_BGR2GRAY)
        transform = transforms.Compose([
            transforms.Resize((arg.img_size)),
            transforms.ToTensor(),
        ])
        next_image = cv.resize(next_image[:288, :int(512*0.79)], (arg.img_size, arg.img_size))
        next_image = transform(Image.fromarray(np.uint8(next_image)))
        next_image = next_image.to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        # current_pred = pred
        # next_pred = model(next_image)

        # add the pair to buffer (Replay Buffer)
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > arg.replay_memory_size:
            del replay_memory[0]  # delete pairs out of limit

        # random sample training data from buffer
        batch = random.sample(replay_memory, min(len(replay_memory), arg.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        # unpack batches
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        # put these batchs into gpu or cpu
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        current_prediction_batch = model(state_batch)  # do inference on current state
        next_prediction_batch = model(next_state_batch) # get the prediction of action of next state

        y_batch = torch.cat(
            tuple(reward if terminal else reward + arg.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))  # set y to reward if dead, or to nest_states's q value times gamma aff reward

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)  # calculate Q value
        optimizer_ft.zero_grad()  # set gradient as zero
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)  # calculate loss
        loss.backward()   # back prop to calculate the gradient
        optimizer_ft.step()  # use gradient descent to update the parameters of the model

        state = next_state  # update state
        Loss_list.append(loss)  # save this epoch's loss to memory

        print("Epoch: {}/{}, {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            epoch + 1,
            arg.num_epoch,
            ["KEEP", "JUMP"][action],
            loss,
            epsilon_decrements[epoch],
            reward,
            torch.max(pred)))  # Print the training log

    torch.save(model.state_dict(), "./model_save/flappy_bird.pt")  # save the well trained model

    # show the loss curve
    x = range(0, num_epochs)
    y = Loss_list
    plt.subplot(2, 1, 2)
    plt.plot(x, y, '.-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.show()
    plt.close()



##########


if __name__ == '__main__':
    arg = args()  # get command line parameters

    # model = models.resnet18()  # load resnet18 from pytorch
    # model = models.mobilenet_v3_small()
    # model.features[0][0] = nn.Conv2d(arg.num_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # # # model.conv1 = nn.Conv2d(arg.num_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
    # # #                         bias=False)  # change the input img channel size at the first conv layer
    # model.classifier[3] = nn.Linear(1024, arg.num_class)  # The last layer of the nn, which output num_class values
    # model.fc = nn.Linear(512, arg.num_class)  # The last layer of the nn, which output num_class values
    model = Net()

    optimizer_ft = optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9)  # SGD optimizer
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5)

    criterion = nn.MSELoss()  # MSE loss function
    num_epochs = arg.num_epoch  # number of epoch

    model_ft = train(model, criterion, optimizer_ft, num_epochs)  # call the training function
