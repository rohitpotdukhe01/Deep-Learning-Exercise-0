import json
import collections
import matplotlib.pyplot as plt
import random
import numpy as np
import skimage.transform


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
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        with open(self.label_path) as f:
            data1 = json.load(f)
        count = len(data1)
        ims = []
        lbl = []
        if (self.batch_epoch >= count - 1):
            self.current_ep += 1
            self.batch_epoch = 0
        h = self.batch_epoch

        def split_dict(d, n, h):
            keys = list(d.keys())
            for i in range(0, 1):
                yield {k: d[k] for k in keys[h:h + n]}

            # data =sorted(data1.items(), key=lambda x: x[1])

        if self.shuffle:
            data = collections.OrderedDict(sorted(data1.items()))

        if self.shuffle == 'false':
            data = collections.OrderedDict(data1.items())

        for item in split_dict(data, self.batch_size, h):
            images_data = list(item.keys())
            labels_data = list(item.values())
            length = len(images_data)
            # Iterating the index
            # same as 'for i in range(len(list))'

            for x in images_data:
                img = np.load(self.file_path + x + '.npy')
                img = skimage.transform.resize(img, self.image_size)
                img = self.augment(img)
                ims.append(img)
                index = images_data.index(x)
                lbl.append(self.class_dict[labels_data[index]])
            h += 1
            images = np.array(ims)
            labels = np.array(lbl)
            self.batch_epoch += self.batch_size
        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        functions = [lambda x: x, np.rot90, np.flip]
        img = random.choice(functions)(img)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_ep

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        batch = self.next()
        rows = int(np.ceil(len(batch[0]/3)))
        cols = 3 
        titles = []
        for y in batch[1]:
            titles.append(self.class_name(batch[1]))
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
        
        i = j = 1
            #print(y)
        for x in batch[0]:
            if j > cols:
                j = 1
                i += 1
            
            fig.add_trace(go.Image(z = x * 255), i, j)
            fig.update_xaxes(visible=False, showticklabels=False)
            fig.update_yaxes(visible=False, showticklabels=False)
            j += 1
            
        fig.show()              
