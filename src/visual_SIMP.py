import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

# Load your data
data1 = sio.loadmat('u_data.mat')
data2 = sio.loadmat('rho_data.mat')

u_data = data1['arr']
rho_data = data2['arr']

# Load your mesh (nodes and elements)
mat_data1 = sio.loadmat("updated_mat_file.mat")

# Extract nodes and elements
nodes = mat_data1['nodes']
elem = mat_data1['elements']

shp = rho_data.shape

def plot_rho(nodes, elem, rho):
    patches_list = []
    color_values = []

    for el in range(len(elem)):
        n1, n2, n3, n4 = elem[el, 1:].astype(int)  # Get the node indices for the element
        x_coords = nodes[[n1, n2, n3, n4], 1]
        y_coords = nodes[[n1, n2, n3, n4], 2]

        polygon = patches.Polygon(xy=list(zip(x_coords, y_coords)), closed=True)
        patches_list.append(polygon)
        color_values.append(rho[el])  # Assuming rho is defined per element

    fig, ax = plt.subplots()
    p = PatchCollection(patches_list, cmap='cividis', edgecolor='none')  # Use Greys for black-and-white
    p.set_array(np.array(color_values))
    ax.add_collection(p)
    plt.colorbar(p, ax=ax, label='Density')
    ax.set_xlim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_ylim(nodes[:, 2].min(), nodes[:, 2].max())
    ax.set_aspect('equal')
    plt.title('Density Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Function to plot u (assuming u is a vector field with 2 components per node)
def plot_u(nodes, elem, u):
    x = nodes[:, 1]
    y = nodes[:, 2]

    # Decompose u into its components
    ux = u[:, 0]  # x-component of displacement
    uy = u[:, 1]  # y-component of displacement

    plt.figure()
    plt.quiver(x, y, ux, uy)
    plt.title('Displacement Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.show()

# Iterate through your data and plot
for i in range(0, shp[0], 5):
    rho = rho_data[i]
    u = u_data[i]

    print(i, np.sum(rho))  # Adjust this line if necessary to match your previous output expectations

    # Plot rho and u
    plot_rho(nodes, elem, rho)
    plot_u(nodes, elem, u)
