import os
from PIL import Image

my_path = "C:/Users/fawaz/Documents/GitHub/Music-Generator-AI/audio_images_raw"
my_end_path = "C:/Users/fawaz/Documents/GitHub/Music-Generator-AI/audio_images_cropped"
all_files = os.listdir(my_path)

left = 0
top = 360
right = 430
bottom = 513

background_sample = Image.open("background_sample.png")

i = 0

for image in all_files:
    link = my_path + "/" + image
    
    i += 1
    if( i % 100 == 0):
        print(str(float(i) / len(all_files)) + '%')

    img = Image.open(link)
    img_size = img.size

    width = img_size[0]
    height = img_size[1]

    if(width < 430):
        background = background_sample.copy()
        background.paste(img, (0, 0))
        img_cropped = background.crop((left, top, right, bottom))
    else:
        img_cropped = img.crop((left, top, right, bottom))

    img_cropped.save(my_end_path + "/" + image)

    # exit()
