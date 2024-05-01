import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import skimage
# import dicom
from skimage import color
from skimage import io
from skimage import filters

from math import fabs, sqrt
import threading

# Read and Pre-process image
img_name = "images/Lenna.png"
image = mpimg.imread(img_name)
rgbImage = mpimg.imread(img_name)
image = color.rgb2gray(image)
edges = filters.scharr(image)

pixels = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
# calculate laplacian filter and gradient magnitude
maxGradient = [0]

def edge_detection(pixels, image, maxGradient, start, end):
    for row in range(start, end):
        for col in range(1, len(image[row]) - 1):
            pixels[row][col] = ((1 * image[row-1][col-1]) + (4 * image[row-1][col]) + (1 * image[row-1][col+1]) + (4 * image[row][col-1]) + (-20 * image[row][col]) + (4 * image[row][col+1]) + (1 * image[row+1][col-1]) + (4 * image[row+1][col]) + (1 * image[row+1][col+1]))/6
            maxGradient[0] = max(maxGradient[0], pixels[row][col])

t1 = threading.Thread(target=edge_detection, args=(pixels, image, maxGradient, 1, int(len(image)/3)))
t2 = threading.Thread(target=edge_detection, args=(pixels, image, maxGradient, int(len(image)/3), int(2 * len(image)/3)))
t3 = threading.Thread(target=edge_detection, args=(pixels, image, maxGradient, int(2 * len(image)/3), len(image) - 1))
t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()
print(maxGradient[0])

for row in range(1, len(image)):
    for col in range(1, len(image[row])):
        pixels[row][col] = 1 - (pixels[row][col]/maxGradient[0])
        if pixels[row][col] < 1.2:
            pixels[row][col] = 0


# Convert image to graph
G = {}
maxD = 0

for row in range(0, len(pixels)):
    for col in range(0, len(pixels[row])):
        if row == 0 or col == 0 or row >= len(pixels) - 1 or col >= len(pixels[row]) - 1:
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
                    dist[n] = fabs(pixels[row][col + 1] - pixels[row - 1][col])/sqrt(2)
                # top left
                elif row > n[0] and col > n[1]:
                    dist[n] = fabs(pixels[row][col - 1] - pixels[row - 1][col])/sqrt(2)
                # bottom right
                elif row < n[0] and col < n[1]:
                    dist[n] = fabs(pixels[row][col + 1] - pixels[row + 1][col])/sqrt(2)
                # bottom left
                else:
                    dist[n] = fabs(pixels[row][col - 1] - pixels[row + 1][col])/sqrt(2)
                    

            # horizontal
            elif col != n[1]:
                # ->
                if col < n[1]:
                    dist[n] = fabs((pixels[row - 1][col] + pixels[row - 1][col + 1])/2 - (pixels[row + 1][col] + pixels[row + 1][col + 1])/2)/2
                # <-
                else:
                    dist[n] = fabs((pixels[row - 1][col] + pixels[row - 1][col - 1])/2 - (pixels[row + 1][col] + pixels[row + 1][col - 1])/2)/2
            # vertical
            elif row != n[0]:
                # up
                if row > n[0]:
                    dist[n] = fabs((pixels[row][col - 1] + pixels[row - 1][col - 1])/2 - (pixels[row][col + 1] + pixels[row - 1][col + 1])/2)/2
                # down
                else:
                    dist[n] = fabs((pixels[row][col - 1] + pixels[row + 1][col - 1])/2 - (pixels[row][col + 1] + pixels[row + 1][col + 1])/2)/2
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
    plt.draw()

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
plt.imshow(rgbImage)
plt.autoscale(False)
plt.title('Live-Wire Tool')
plt.show()