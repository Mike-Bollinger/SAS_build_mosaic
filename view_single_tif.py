import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
from rasterio.windows import Window

def view_geotiffs_simple(image_paths):
    """
    Reads and displays multiple GeoTIFFs on a single plot, honoring their affine transformations.

    Args:
        image_paths (list): A list of paths to the GeoTIFF files.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    for image_path in image_paths:
        try:
            with rasterio.open(image_path) as src:
                affine = src.transform
                print(f"Affine a (width of a pixel in the x-direction): {affine.a}")
                print(f"Affine b (shear parameter): {affine.b}")
                print(f"Affine c (This is the x-coordinate (e.g., Easting or Longitude) of the top-left corner of the top-left pixel of your image): {affine.c}")
                print(f"Affine d (shear parameter): {affine.d}")
                print(f"Affine e (height of a pixel in the y-direction): {affine.e}")
                print(f"Affine f (y-coordinate (e.g., Northing or Latitude) of the top-left corner of the top-left pixel.): {affine.f}")
                print(f"CRS: {src.crs}")
                print(f"Image dimensions (width x height): {src.width} x {src.height}")
                print(f"Image shape: {src.shape}")
                print(f"Number of bands: {src.count}")

                image_data = src.read()

                bounds = src.bounds
                extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
                
                ax.imshow(image_data[0], extent=extent, origin='lower')

        except FileNotFoundError:
            print(f"Warning: The file was not found at {image_path} and will be skipped.")
        except Exception as e:
            print(f"An error occurred with {image_path}: {e}")

    # Improve the plot
    ax.set_title("Multiple GeoTIFF Display")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', color='grey', alpha=0.6)
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()

if __name__ == '__main__':

    geotiff_paths = [
        r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\crop\KRAKEN - 2025-09-06T22-57-51 - HBL - port - 3_10cm_crop.tif",
        r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\crop\KRAKEN - 2025-09-06T22-57-51 - HBL - port - 4_10cm_crop.tif",
        r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\crop\KRAKEN - 2025-09-06T22-57-51 - HBL - port - 5_10cm_crop.tif",
        r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\crop\KRAKEN - 2025-09-06T22-57-51 - HBL - port - 6_10cm_crop.tif",
    ]
    
    view_geotiffs_simple(geotiff_paths)
