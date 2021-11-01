from giffer import make_gif
import os
from os.path import dirname, join

path = dirname(__file__)
img_folder = join(path, "img")
save_as = join(path, "k_means.gif")
temp_folder = join(path, "temp")


# for elem in os.listdir(temp_folder):
#     os.remove(join(temp_folder, elem))

# os.rmdir(temp_folder)