'''
Load PNG image
'''
import imageio as iio
import matplotlib.pyplot as plt

im = iio.imread('patterns/Sprite-0001.png')  # read an image
print(im.shape)  # im is a NumPy array of shape 

plt.imshow(im)
plt.show()
