import numpy as np 
import pandas as pd
import plotly.express as px 

df_train = pd.read_csv('../input/fashion-mnist_train.csv')
X_train = df_train.drop('label', axis=1).to_numpy().reshape((70000, 28, 28))
columns = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
y_train = pd.get_dummies(df_train['label']).rename(columns=columns).to_numpy()

def adjacent_pixels(pixel_cord: tuple):
    x = pixel_cord[0]
    y = pixel_cord[1]
    cord_matrix = [[max(0, x - 1), max(0, y - 1)], [max(0, x - 1), y], [max(0, x - 1), min(27, y + 1)], 
                    [x, max(0, y - 1)], [x, y], [x, min(27, y + 1)], 
                    [min(27, x + 1), max(0, y - 1)], [min(27, x + 1), y], [min(27, x + 1), min(27, y + 1)]]
    return cord_matrix

def similarity_score(inputs, edge_map):
    return (np.multiply(inputs, edge_map).sum())

def edge_map(image_class):
    edge_map_image_class = np.zeros((26, 26))
    for image in skewed_distribution[image_class]:
        edge_map_image_class = np.add(edge_map_image_class, np.multiply(image, 1/7000))
    return edge_map_image_class

all_differences = []
difference_sums = []
coordinates = [[j, i] for j in range(1, 27) for i in range(1, 27)]
  
for image in X_train:
    for (x, y) in coordinates:
        arr = adjacent_pixels((x, y))
        minuend = np.full((3, 3), image[x][y])
        subtrahend = []
        for cord in arr:
            subtrahend.append(image[cord[0]][cord[1]])
        subtrahend = np.array(subtrahend).reshape((3, 3))
        difference = np.subtract(minuend, subtrahend).sum()
        difference_sums.append(difference)
    all_differences.append(difference_sums)
    difference_sums = []

skewed_distribution = np.array(list(map(lambda x: abs(x) if abs(x) >= 300 else -1 * (300 - abs(x)), np.array(all_differences).flatten()))).reshape(70000, 26, 26)

coordinates = [[j, i] for j in range(1, 27) for i in range(1, 27)]
X = []
for image in X_train:
    for (x, y) in coordinates:
        X.append(image[x][y])
X_train = np.array(X).reshape((70000, 26, 26))

class_indices = {0: [], 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
for index, array in enumerate(y_train):
    class_indices[np.where(array == 1)[0].item()].append(index)
t_shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot = list(class_indices.values())

class_list = [edge_map(image_class) for image_class in [t_shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, boot]]
similarity = [similarity_score(skewed_distribution[11], weights) for weights in class_list] # Replace `skewed_distribution[11]` with desired test image
print(columns[np.where(similarity == max(similarity))[0].item()])