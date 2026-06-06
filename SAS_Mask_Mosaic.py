import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

# === CONFIGURATION ===
base_dir = Path(r"C:\Users\Map.AtSea\Documents\EN2501")
rgba_path = base_dir / "EN2501_Dive036_10cm_mosaic24 bit.tif"
main_tif_path = base_dir / "EN2501_Dive036_10cm_mosaic.tif"

dataset_name = main_tif_path.stem
clipped_output = base_dir / f"{dataset_name}_Band1_clipped.tif"
projected_output = base_dir / f"{dataset_name}_Band1_3857.tif"

# === STEP 1: Build valid data mask from RGBA raster ===
print("🔍 Reading RGBA raster and building data mask...")

with rasterio.open(rgba_path) as src:
    r, g, b, a = [src.read(i) for i in (1, 2, 3, 4)]
    rgba_mask = ~((r == 0) & (g == 0) & (b == 0) & ((a == 0) | (a == 255)))
    mask_transform = src.transform
    mask_crs = src.crs

# === STEP 2: Read Band 1 and apply mask ===
print("✂️ Extracting Band 1 and applying mask...")

with rasterio.open(main_tif_path) as src:
    band1 = src.read(1)  # Band 1 only
    profile = src.profile

    if (src.width != r.shape[1]) or (src.height != r.shape[0]):
        raise ValueError("RGBA raster and main raster must have same shape!")

    # Apply the mask: set NoData (0) where invalid
    band1_masked = np.where(rgba_mask, band1, 0)

    # Update profile for single band output
    profile.update({
        "count": 1,
        "dtype": band1_masked.dtype,
        "nodata": 0
    })

    with rasterio.open(clipped_output, "w", **profile) as dst:
        dst.write(band1_masked, 1)

print(f"✅ Clipped raster written to: {clipped_output}")

# === STEP 3: Reproject to EPSG:3857 ===
print("🗺️ Reprojecting to EPSG:3857...")

dst_crs = 'EPSG:3857'
with rasterio.open(clipped_output) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(projected_output, 'w', **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

print(f"✅ Reprojected raster saved to: {projected_output}")