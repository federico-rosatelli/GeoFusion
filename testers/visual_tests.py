from src.io import loader as data_loader
from src.visualization import plotting as shape_visual

def plasma_bound_visual():
    dataset = data_loader.load_constellaration_dataset()
    sample_config = dataset[7]


    surface_data = shape_visual.get_surface_coordinates(sample_config)
    X, Y, Z = surface_data['X'], surface_data['Y'], surface_data['Z']
    shape_visual.plot_stellarator_shape(X, Y, Z)
