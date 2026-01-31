import numpy as np
import matplotlib.pyplot as plt
import os
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



def plot_loss(metric_name, loss_history):
    
    try:
        
        plt.figure(figsize=(8, 5))
        
        plt.plot(range(1, len(loss_history) + 1), loss_history, 'o-', color='#d62728', linewidth=2, markersize=5, label='MSE Loss')

        plt.title(f"Training Loss: {metric_name.upper()}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        plt.tight_layout()
        
        plot_dir = "public/images"
        filename = f"{metric_name}_loss_plot.png"
        file_path = os.path.join(plot_dir, filename)
        plt.savefig(file_path, dpi=300)
        
    except Exception as e:
        print(f"Error loading or plotting loss data: {e}")
