'''
Coords for small CLP
'''
import numpy as np

import matplotlib.pyplot as plt

#parameters
length = 13
padding = 8
num_classes = 3
spc = 2 #spacing between points

radius = length*spc/np.pi
radius_int = int(radius+0.5) #round and pad
print("radius", radius)

N = int(2.*radius+padding+0.5)
print("N:", N)

#===========
#network architecture
offset = [N//2-(radius_int+padding)//2, N//2+radius_int//2]
slopes = [[0,-spc], [1,-spc], [spc,-1], [spc,0], [spc,1], [1,spc], [0,spc]]
steps = 2
#input coords
input_coords = []
#draw anti-clockwise
current = offset
input_coords.append(current)
for vec in slopes:
    for step in range(steps):
        new_point = [current[0]+vec[0], current[1]+vec[1]]
        input_coords.append(new_point)
        current = new_point
print("num points:", len(input_coords))
print(input_coords)

#output
output_coords = []

cells = np.zeros((N,N))
for point in input_coords:
    cells[point[0], point[1]] = 255

plt.imshow(cells)
plt.title("Initial Architecture")
plt.savefig("vnn_clp_coords.png", dpi=150)
plt.show()
