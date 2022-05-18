# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import json
import collections
import matplotlib.pyplot as plt
import random
import numpy as np

h = 0
class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

def split_dict(d, n, h):
    keys = list(d.keys())
    for i in range(0, 1):
        yield {k: d[k] for k in keys[h:h + 3]}


with open('/Users/chitraahuja/Downloads/exercise0_material 2/src_to_implement/data/Labels.json') as f:
    data1 = json.load(f)
    # data =sorted(data1.items(), key=lambda x: x[1])
    data = collections.OrderedDict(sorted(data1.items()))

    folder_dir = '/Users/chitraahuja/Downloads/exercise0_material 2/src_to_implement/data/exercise_data'

for item in split_dict(data, 3, h):
    images_data = list(item.keys())
    labels_data = list(item.values())
    length = len(images_data)
    # Iterating the index
    # same as 'for i in range(len(list))'

    for x in images_data:
        img = np.load(folder_dir + '/' + x + '.npy')
        plt.imshow(img, cmap='gray')
        functions = [lambda x: x, np.rot90, np.flip]
        res=random.choice(functions)(img)
        plt.imshow(res)
        index = images_data.index(x)
        print(class_dict[labels_data[index]])
        plt.show()

    print(item)
    h = h + 3





