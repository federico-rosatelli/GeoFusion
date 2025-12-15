import numpy as np
import matplotlib.pyplot as plt
from src.io.loader import load_constellaration_dataset

from src.physics.geometry import get_surface_coordinates

def plot_stellarator_shape(X, Y, Z):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=4, cstride=4, alpha=0.9, antialiased=True)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


    ax.set_title("Plasma Boundary")
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    

    plt.show()

if __name__ == "__main__":

    dataset = load_constellaration_dataset()
    
    if dataset:
       
        sample_config = dataset[0]

    
    surface_data = get_surface_coordinates(sample_config)
    X, Y, Z = surface_data['X'], surface_data['Y'], surface_data['Z']
    plot_stellarator_shape(X, Y, Z)