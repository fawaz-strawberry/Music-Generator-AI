import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from NetworkArchitectureAudio import Discriminator, Generator, initialize_weights
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#HyperParams
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_AUDIO_IMAGES = 100
FILE_PATH = "./GeneratedAudio/"
FINAL_FILE_PATH = "./FinalGeneratedAudio/"

LEARNING_RATE = 2e-4
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3

NOISE_DIM = 10
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
LOAD_PARAMS = True

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint") 
    checkpoint = torch.load(checkpoint)
    
    gen.load_state_dict(checkpoint['gen_state_dict'])
    opt_gen.load_state_dict(checkpoint['gen_opt'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    opt_disc.load_state_dict(checkpoint['disc_opt'])

#This function will rebuild the full image
def fix_image(file_name):
    background_sample = Image.open("background_sample.png")

    img = Image.open(file_name)

    image_size = img.size
    back_size = background_sample.size

    background = background_sample.copy()
    background.paste(img, (0, back_size[1] - image_size[1]))

    background.save(FINAL_FILE_PATH + file_name)

#This function will generate a new track with a randomized noise path
def generate_image(my_noise, file_name):
    with torch.no_grad():
        fake = gen(my_noise)

        images = fake.cpu().numpy()
        final_image = np.squeeze(images[0]) * 255
        im = Image.fromarray(final_image)


        if(im.mode != "L"):
            im = im.convert('L')
        im.save(FILE_PATH + file_name)



gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))  

fixed_noise = torch.randn(1, Z_DIM, 1, 1).to(device)

load_checkpoint("my_checkpoint.pth.tar")

for i in range(NUM_AUDIO_IMAGES):
    fixed_noise = torch.randn(1, Z_DIM, 1, 1).to(device)
    generate_image(fixed_noise, "output_" + str(i) + ".png")
    fix_image("output_" + str(i) + ".png")

# generate_image(fixed_noise, "0_0.png")

