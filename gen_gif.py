#!/usr/bin/env python
# coding=utf-8
# __author__='Alfred'

import imageio as io
import glob

img_dir = r'D:\workspace\my_github\Car-Behavioral-Cloning\gif_images\*.jpg'

images = list(map(lambda i: io.imread(i), glob.glob(img_dir)))

io.mimsave('run2.gif', images, fps=60)
