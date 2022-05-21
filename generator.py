import json
import os
import random
import numpy as np
import skimage.transform
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    batch_epoch = current_ep = 0

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.batch_epoch = 0
        self.index = 0
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size

        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        # TODO: implement constructor

    def next(self):
        with open(self.label_path) as f:
            jsonData = json.load(f)
        pth, dirs, imgFiles = next(os.walk(self.file_path))
        count = len(imgFiles)
        batch_imgs = []
        batch_lbls = []
        if self.index >= count - 1:
            self.current_ep += 1
            self.batch_epoch = 0
        batch_data = []
        keys = list(jsonData.keys())
        temp_index = self.index
        for j in range(self.index, self.batch_size + self.index):
            if temp_index > count - 1:
                temp_index = 0
                self.batch_epoch += 1
            if self.shuffle:
                random.shuffle(keys)
            batch_data.append({keys[temp_index]: jsonData[keys[temp_index]]})
            temp_index += 1

        for item in batch_data:
            images_data = list(item.keys())
            labels_data = list(item.values())
            for x in images_data:
                img = np.load(self.file_path + x + '.npy')
                img = skimage.transform.resize(img, self.image_size)
                img = self.augment(img)
                batch_imgs.append(img)
                index = images_data.index(x)
                batch_lbls.append(labels_data[index])
        images = np.array(batch_imgs)
        labels = np.array(batch_lbls)
        self.index += self.batch_size
        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        if self.mirroring and random.randint(0, 1) == 0:
            img = np.flip(img, 1)
        if self.rotation and random.randint(0, 1) == 0:
            img = np.rot90(img, random.randint(0, 3))

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]

    def current_epoch(self):
        # return the current epoch number
        return self.current_ep

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        batch = self.next()
        rows = int(np.ceil(len(batch[0] / 3)))
        cols = 3
        titles = []
        for y in batch[1]:
            titles.append(self.class_name(y))
        img_with_label = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        i = j = 1
        for x in batch[0]:
            if j > cols:
                j = 1
                i += 1

            img_with_label.add_trace(go.Image(z=x * 255), i, j)
            img_with_label.update_xaxes(visible=False, showticklabels=False)
            img_with_label.update_yaxes(visible=False, showticklabels=False)
            j += 1

        img_with_label.show()
