import torch
from torch import nn
import torch.optim as optim
import numpy as np
import math

from PIL import Image


#Generator Network
class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(input_size, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.l2 = nn.Linear(64, 256)
        self.l3 = nn.Linear(256, 1025)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        out = self.l1(input_tensor)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.sigmoid(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.leaky_relu = nn.LeakyReLU()
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        out = self.l1(input_tensor)
        out = self.leaky_relu(out)
        out = self.l2(out)
        out = self.leaky_relu(out)
        out = self.l3(out)
        out = self.sigmoid(out)
        return out

def drawImage(input_array, width, height, filename):
    temp_list = []
    for i in range(width):
        for j in range(height):
            my_val = input_array[i * height + j]
            temp_list.append((int(255 * my_val), int(255 * my_val), int(255 * my_val)))

    img = Image.new('RGB', (width, height))
    img.putdata(temp_list)
    img.save(filename)

inputs = np.arange(1, 17, 1)
final_inputs = torch.rand(len(inputs), 10, dtype=torch.float64)
outputs = np.zeros((len(inputs), 1025))
for i, input in enumerate(inputs):
    print("i: " + str(i) + ", ")
    
    for j in range(32):
        for k in range(32):
            x_val = k - 16
            y_val = 16 - j

            if(int(math.sqrt(x_val**2 + y_val**2)) == input):
                outputs[i][(32 * j) + k] = 1

            #if( == )
    outputs[i][32*32] = (i/17)
    #r^2 = (x-a)^2 + (y-b)^2
    #32x32

num_epochs = 2000

print(len(final_inputs[0]))
print(len(outputs[0]))
generator = Generator(len(final_inputs[0]))
discriminator = Discriminator(len(outputs[0]))
discriminator.double()
generator.double()

learning_rate = .0002

criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(generator.parameters(), lr=.00002, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.000002, betas=(0.5, 0.999))


torch.set_default_dtype(torch.float64)
radiuses = final_inputs
circles = torch.from_numpy(outputs)


D_labels = torch.ones(1, 1)
D_fakes = torch.zeros(1, 1)

m = 0
for epoch in range(num_epochs):
    for i in range(len(radiuses)):
        #Train Discriminator
        training_guess = discriminator.forward(circles[i])
        D_x_loss = criterion(training_guess, D_labels)

        training_gen = discriminator.forward(generator(radiuses[i]))
        D_z_loss = criterion(training_gen, D_fakes)
        
        D_loss = D_x_loss + D_z_loss


        discriminator.zero_grad()
        D_loss.backward()

        D_optimizer.step()


        training_gen = discriminator.forward(generator(radiuses[i]))
        G_loss = criterion(training_gen, D_labels)
        generator.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    if ((epoch % 10) == 0):
        output = generator.forward(radiuses[10])
        out = output.detach().numpy()
        drawImage(out, 32, 32, ("circle_" + str(m) + ".png"))
        m += 1
        

    print('Epoch: {}/{}, D Loss: {}, G Loss: {}'.format(epoch, num_epochs, D_loss.item(), G_loss.item()))

        
# for i in range(4, 5):
#     output = generator.forward(radiuses[i])
#     out = output.detach().numpy()
#     drawImage(outputs[i], 32, 32)
#     drawImage(out, 32, 32)
