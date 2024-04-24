import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import skimage
# import dicom
from skimage import color
from skimage import io
from skimage import filters

from math import fabs, sqrt

# Read and Pre-process image
img_name = "Lenna.png"
image = mpimg.imread(img_name)
image = color.rgb2gray(image)
edges = filters.scharr(image)

# Convert image to graph
G = {}
maxD = 0
for row in range(0, len(image)):
    for col in range(0, len(image[row])):
        if row == 0 or col == 0 or row >= len(image) - 1 or col >= len(image[row]) - 1:
            G[(row,col)] = {}
            continue
        
        neighbors = []
      
        neighbors.append( (row-1, col) )
        neighbors.append( (row+1, col) )
        neighbors.append( (row, col-1) )
        neighbors.append( (row, col+1) )
        neighbors.append( (row-1, col-1) )
        neighbors.append( (row-1, col+1) )
        neighbors.append( (row+1, col-1) )
        neighbors.append( (row+1, col+1) )
        
        dist = {}
        for n in neighbors:
            # cost function (can be replaced)

            # diaganol
            if row != n[0] and col != n[1]:
                # top right
                if row > n[0] and col < n[1]:
                    dist[n] = fabs(image[row][col + 1] - image[row - 1][col])/sqrt(2)
                # top left
                elif row > n[0] and col > n[1]:
                    dist[n] = fabs(image[row][col - 1] - image[row - 1][col])/sqrt(2)
                # bottom right
                elif row < n[0] and col < n[1]:
                    dist[n] = fabs(image[row][col + 1] - image[row + 1][col])/sqrt(2)
                # bottom left
                else:
                    dist[n] = fabs(image[row][col - 1] - image[row + 1][col])/sqrt(2)
                    

            # horizontal
            elif col != n[1]:
                # ->
                if col < n[1]:
                    dist[n] = fabs((image[row - 1][col] + image[row - 1][col + 1])/2 - (image[row + 1][col] + image[row + 1][col + 1])/2)/2
                # <-
                else:
                    dist[n] = fabs((image[row - 1][col] + image[row - 1][col - 1])/2 - (image[row + 1][col] + image[row + 1][col - 1])/2)/2
            # vertical
            elif row != n[0]:
                # up
                if row > n[0]:
                    dist[n] = fabs((image[row][col - 1] + image[row - 1][col - 1])/2 - (image[row][col + 1] + image[row - 1][col + 1])/2)/2
                # down
                else:
                    dist[n] = fabs((image[row][col - 1] + image[row + 1][col - 1])/2 - (image[row][col + 1] + image[row + 1][col + 1])/2)/2
            maxD = max(maxD, dist[n])
            
        G[(row,col)] = dist

for key in G:
    for n in G[key]:
        if key[0] != n[0] and key[1] != n[1]:
            G[key][n] = (maxD - G[key][n]) * sqrt(2)
        else:
            G[key][n] = (maxD - G[key][n])


# Apply Dijkstra's Algorithm
from dijkstra import shortestPath

INTERACTIVE = True
from itertools import cycle
import numpy as np
COLORS = cycle('rgbyc')

start_point = None
# current_color = COLORS.next()
current_color = next(COLORS)
current_path = None
length_penalty = 10.0

def button_pressed(event):
    global start_point
    if start_point is None:
        start_point = (int(event.ydata), int(event.xdata))
        
    else:
        end_point = (int(event.ydata), int(event.xdata))
        path = shortestPath(G, start_point, end_point, length_penalty=length_penalty)
        print(path)
        plt.plot(np.array(path)[:,1], np.array(path)[:,0], c=current_color)
        start_point = end_point

def mouse_moved(event):
    if start_point is None:
        return
    
    end_point = (int(event.ydata), int(event.xdata))
    path = shortestPath(G, start_point, end_point, length_penalty=length_penalty)
    
    global current_path
    if current_path is not None:
        current_path.pop(0).remove()
    current_path = plt.plot(np.array(path)[:,1], np.array(path)[:,0], c=current_color)

def key_pressed(event):
    if event.key == 'escape':
        global start_point, current_color
        start_point = None
        # current_color = COLORS.next()
        current_color = next(COLORS)

        global current_path
        if current_path is not None:
            current_path.pop(0).remove()
            current_path = None
            plt.draw()

plt.connect('button_release_event', button_pressed)
if INTERACTIVE:
    plt.connect('motion_notify_event', mouse_moved)
plt.connect('key_press_event', key_pressed)

plt.gray()
plt.imshow(image)
plt.autoscale(False)
plt.title('Live-Wire Tool')
plt.show()