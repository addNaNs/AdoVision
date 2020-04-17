import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf

digit_dirs = [os.path.join('../data/japanesehandwrittendigits/0'+ str(i)) for i in range(10)]

print('total_images = ' + str(len(os.listdir(digit_dirs[0]))))

digit_files = [os.listdir(digit_dirs[i]) for i in range(10)]
print(digit_files[0][:10])


want_to_print = False
if want_to_print:
    for i, img_path in enumerate([os.path.join(digit_dirs[6], f_name) for f_name in digit_files[6][0:5]]):
        print(img_path)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('Off')
        plt.show()
