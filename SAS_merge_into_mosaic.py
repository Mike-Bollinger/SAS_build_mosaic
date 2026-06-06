"""
SAS TIFF Mosaic Builder and Tile Generator

This script takes all TIFF files from a specified directory, merges them into a single mosaic,
and then breaks that mosaic into tile packages that can be read into GIS applications.

Author: GitHub Copilot
Date: September 2025
"""

import os
import glob
import math
import numpy as np
import rasterio
from rasterio import merge, windows
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
from rasterio.profiles import default_gtiff_profile
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class TiffMosaicProcessor:
    """
    A class to handle TIFF file merging and tile generation for GIS applications.
    """
    
    def __init__(self, input_directory: str, output_directory: str, tile_size: int = 1024):
        """
        Initialize the mosaic processor.
        
        Args:
            input_directory (str): Directory containing input TIFF files
            output_directory (str): Directory to save output files
            tile_size (int): Size of tiles in pixels (default: 1024)
        """
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.tile_size = tile_size
        self.mosaic_path = None
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
    def find_tiff_files(self) -> List[str]:
        """
        Find all .tif and .tiff files in the input directory.
        
        Returns:
            List[str]: List of TIFF file paths
        """
        tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        tiff_files = []
        
        for pattern in tiff_patterns:
            files = list(self.input_directory.glob(pattern))
            tiff_files.extend([str(f) for f in files])
            
        print(f"Found {len(tiff_files)} TIFF files in {self.input_directory}")
        return sorted(tiff_files)
    
    def inspect_tiff_files(self, tiff_files: List[str]) -> None:
        """
        Inspect TIFF files to understand their properties and identify flipped rasters.
        
        Args:
            tiff_files (List[str]): List of TIFF file paths
        """
        print("\n" + "="*60)
        print("TIFF FILE INSPECTION")
        print("="*60)
        
        flipped_count = 0
        
        for i, tiff_file in enumerate(tiff_files[:5]):  # Show first 5 files
            try:
                with rasterio.open(tiff_file) as src:
                    print(f"\nFile {i+1}: {Path(tiff_file).name}")
                    print(f"  Dimensions: {src.width} x {src.height}")
                    print(f"  Bands: {src.count}")
                    print(f"  Data type: {src.dtypes[0]}")
                    print(f"  CRS: {src.crs}")
                    print(f"  Bounds: {src.bounds}")
                    
                    # Print affine transformation details
                    affine = src.transform
                    print(f"  Affine transformation:")
                    print(f"    Pixel width (a): {affine.a:.6f}")
                    print(f"    Pixel height (e): {affine.e:.6f}")
                    print(f"    Top-left X (c): {affine.c:.6f}")
                    print(f"    Top-left Y (f): {affine.f:.6f}")
                    
                    # Check if raster is flipped
                    is_flipped = affine.a < 0 or affine.e > 0
                    if is_flipped:
                        flipped_count += 1
                        print(f"    ⚠️  WARNING: This raster appears to be flipped!")
                        if affine.a < 0:
                            print(f"      - Negative pixel width (horizontally flipped)")
                        if affine.e > 0:
                            print(f"      - Positive pixel height (vertically flipped)")
                    else:
                        print(f"    ✓ Raster orientation is normal")
                    
            except Exception as e:
                print(f"Error reading {tiff_file}: {e}")
                
        if len(tiff_files) > 5:
            print(f"\n... and {len(tiff_files) - 5} more files")
            
            # Check remaining files for flipped status
            for tiff_file in tiff_files[5:]:
                try:
                    with rasterio.open(tiff_file) as src:
                        affine = src.transform
                        if affine.a < 0 or affine.e > 0:
                            flipped_count += 1
                except Exception:
                    pass
        
        if flipped_count > 0:
            print(f"\n⚠️  Found {flipped_count} flipped raster(s) out of {len(tiff_files)} total files.")
            print("   These will be automatically normalized during mosaic creation.")
        else:
            print(f"\n✓ All {len(tiff_files)} rasters have normal orientation.")
    
    def normalize_raster_orientation(self, src_file: str, temp_dir: Path) -> str:
        """
        Normalize a raster's orientation to handle flipped rasters.
        
        Args:
            src_file (str): Path to the source TIFF file
            temp_dir (Path): Directory to store temporary normalized files
            
        Returns:
            str: Path to the normalized file
        """
        temp_file = temp_dir / f"normalized_{Path(src_file).name}"
        
        try:
            with rasterio.open(src_file) as src:
                transform = src.transform
                
                # Check if the raster is flipped (negative pixel width or positive pixel height)
                if transform.a < 0 or transform.e > 0:
                    print(f"  Normalizing flipped raster: {Path(src_file).name}")
                    
                    # Read the data
                    data = src.read()
                    
                    # Handle horizontal flipping (negative pixel width)
                    if transform.a < 0:
                        print(f"    - Fixing horizontal flip (negative pixel width)")
                        # Flip the data horizontally
                        data = data[:, :, ::-1]
                        # Create corrected transform
                        new_a = abs(transform.a)
                        new_c = transform.c + transform.a * src.width  # Adjust origin
                    else:
                        new_a = transform.a
                        new_c = transform.c
                    
                    # Handle vertical flipping (positive pixel height)
                    if transform.e > 0:
                        print(f"    - Fixing vertical flip (positive pixel height)")
                        # Flip the data vertically
                        data = data[:, ::-1, :]
                        # Create corrected transform
                        new_e = -abs(transform.e)
                        new_f = transform.f + transform.e * src.height  # Adjust origin
                    else:
                        new_e = transform.e
                        new_f = transform.f
                    
                    # Create the corrected transform
                    new_transform = rasterio.Affine(
                        new_a,        # Pixel width (always positive)
                        transform.b,  # Rotation (keep original)
                        new_c,        # X coordinate of top-left corner (adjusted)
                        transform.d,  # Rotation (keep original) 
                        new_e,        # Pixel height (always negative)
                        new_f         # Y coordinate of top-left corner (adjusted)
                    )
                    
                    print(f"    - Old transform: a={transform.a:.8f}, e={transform.e:.8f}")
                    print(f"    - New transform: a={new_transform.a:.8f}, e={new_transform.e:.8f}")
                    
                    # Create new metadata
                    new_meta = src.meta.copy()
                    new_meta.update({
                        'transform': new_transform,
                        'compress': 'lzw'
                    })
                    
                    # Write normalized file with corrected data and transform
                    with rasterio.open(temp_file, 'w', **new_meta) as dst:
                        dst.write(data)
                    
                    return str(temp_file)
                else:
                    # File is already properly oriented, return original
                    return src_file
                    
        except Exception as e:
            print(f"Warning: Could not normalize {src_file}: {e}")
            return src_file

    def create_mosaic(self, tiff_files: List[str], output_filename: str = "mosaic.tif") -> str:
        """
        Create a mosaic from multiple TIFF files, handling flipped rasters.
        
        Args:
            tiff_files (List[str]): List of TIFF file paths
            output_filename (str): Name of the output mosaic file
            
        Returns:
            str: Path to the created mosaic file
        """
        print(f"\n" + "="*60)
        print("CREATING MOSAIC")
        print("="*60)
        
        mosaic_path = self.output_directory / output_filename
        
        # Create temporary directory for normalized files
        temp_dir = self.output_directory / "temp_normalized"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Normalize all rasters to handle flipped orientations
            print("Checking and normalizing raster orientations...")
            normalized_files = []
            
            for tiff_file in tiff_files:
                try:
                    normalized_file = self.normalize_raster_orientation(tiff_file, temp_dir)
                    normalized_files.append(normalized_file)
                except Exception as e:
                    print(f"Warning: Could not process {tiff_file}: {e}")
            
            if not normalized_files:
                raise ValueError("No valid TIFF files could be processed")
            
            print(f"Successfully processed {len(normalized_files)} TIFF files")
            
            # Step 2: Open all normalized TIFF files
            src_files = []
            for normalized_file in normalized_files:
                try:
                    src = rasterio.open(normalized_file)
                    src_files.append(src)
                except Exception as e:
                    print(f"Warning: Could not open {normalized_file}: {e}")
            
            if not src_files:
                raise ValueError("No valid TIFF files could be opened after normalization")
            
            print(f"Successfully opened {len(src_files)} normalized TIFF files")
            
            # Step 3: Check CRS consistency
            base_crs = src_files[0].crs
            for i, src in enumerate(src_files[1:], 1):
                if src.crs != base_crs:
                    print(f"Warning: File {i+1} has different CRS ({src.crs}) than first file ({base_crs})")
            
            # Step 4: Merge the files
            print("Merging files...")
            mosaic_array, mosaic_transform = merge.merge(
                src_files,
                method='first',  # Use 'first' to take the first valid pixel value
                dtype=src_files[0].dtypes[0]
            )
            
            # Get metadata from the first file
            mosaic_meta = src_files[0].meta.copy()
            mosaic_meta.update({
                'height': mosaic_array.shape[1],
                'width': mosaic_array.shape[2],
                'transform': mosaic_transform,
                'compress': 'lzw'  # Add compression to reduce file size
            })
            
            # Write the mosaic
            print(f"Writing mosaic to: {mosaic_path}")
            with rasterio.open(mosaic_path, 'w', **mosaic_meta) as dest:
                dest.write(mosaic_array)
            
            # Close source files
            for src in src_files:
                src.close()
            
            print(f"Mosaic created successfully!")
            print(f"Mosaic dimensions: {mosaic_array.shape[2]} x {mosaic_array.shape[1]}")
            print(f"Mosaic bounds: {rasterio.open(mosaic_path).bounds}")
            
            self.mosaic_path = str(mosaic_path)
            
            # Clean up temporary files
            print("Cleaning up temporary files...")
            for temp_file in temp_dir.glob("normalized_*.tif"):
                try:
                    temp_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {temp_file}: {e}")
            
            # Remove temp directory if empty
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other issue
            
            return str(mosaic_path)
            
        except Exception as e:
            print(f"Error creating mosaic: {e}")
            # Clean up temporary files on error
            try:
                for temp_file in temp_dir.glob("normalized_*.tif"):
                    temp_file.unlink()
                temp_dir.rmdir()
            except Exception:
                pass
            raise
    
    def create_mosaic_with_warping(self, tiff_files: List[str], output_filename: str = "mosaic.tif") -> str:
        """
        Alternative mosaic creation method that warps all rasters to a common grid.
        Use this if the standard merging fails due to incompatible raster orientations.
        
        Args:
            tiff_files (List[str]): List of TIFF file paths
            output_filename (str): Name of the output mosaic file
            
        Returns:
            str: Path to the created mosaic file
        """
        print(f"\n" + "="*60)
        print("CREATING MOSAIC WITH WARPING")
        print("="*60)
        
        mosaic_path = self.output_directory / output_filename
        
        try:
            # Step 1: Determine common CRS and bounds
            print("Analyzing input files...")
            bounds_list = []
            crs_list = []
            
            for tiff_file in tiff_files:
                try:
                    with rasterio.open(tiff_file) as src:
                        bounds_list.append(src.bounds)
                        crs_list.append(src.crs)
                except Exception as e:
                    print(f"Warning: Could not read {tiff_file}: {e}")
            
            if not bounds_list:
                raise ValueError("No valid TIFF files found")
            
            # Use the most common CRS
            target_crs = max(set(crs_list), key=crs_list.count)
            print(f"Target CRS: {target_crs}")
            
            # Calculate overall bounds
            min_x = min(bounds.left for bounds in bounds_list)
            max_x = max(bounds.right for bounds in bounds_list)
            min_y = min(bounds.bottom for bounds in bounds_list)
            max_y = max(bounds.top for bounds in bounds_list)
            
            print(f"Overall bounds: ({min_x:.6f}, {min_y:.6f}, {max_x:.6f}, {max_y:.6f})")
            
            # Step 2: Determine target resolution (use the highest resolution from input files)
            target_resolution = float('inf')
            for tiff_file in tiff_files:
                try:
                    with rasterio.open(tiff_file) as src:
                        res = min(abs(src.transform.a), abs(src.transform.e))
                        target_resolution = min(target_resolution, res)
                except Exception:
                    pass
            
            if target_resolution == float('inf'):
                target_resolution = 1.0  # Default 1 meter resolution
            
            print(f"Target resolution: {target_resolution}")
            
            # Step 3: Calculate output dimensions
            width = int((max_x - min_x) / target_resolution)
            height = int((max_y - min_y) / target_resolution)
            
            print(f"Output dimensions: {width} x {height}")
            
            # Step 4: Create target transform
            target_transform = rasterio.Affine(
                target_resolution, 0, min_x,
                0, -target_resolution, max_y
            )
            
            # Step 5: Create output array
            with rasterio.open(tiff_files[0]) as sample_src:
                bands = sample_src.count
                dtype = sample_src.dtypes[0]
            
            output_array = np.zeros((bands, height, width), dtype=dtype)
            
            # Step 6: Warp each input file to the target grid
            print("Warping and merging files...")
            for i, tiff_file in enumerate(tiff_files):
                try:
                    with rasterio.open(tiff_file) as src:
                        print(f"  Processing file {i+1}/{len(tiff_files)}: {Path(tiff_file).name}")
                        
                        # Create temporary array for this file
                        temp_array = np.zeros((bands, height, width), dtype=dtype)
                        
                        # Warp this file to the target grid
                        reproject(
                            source=rasterio.band(src, list(range(1, bands + 1))),
                            destination=temp_array,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear
                        )
                        
                        # Merge with output (use first valid pixel)
                        mask = temp_array != 0
                        output_array[mask] = temp_array[mask]
                        
                except Exception as e:
                    print(f"  Warning: Could not process {tiff_file}: {e}")
            
            # Step 7: Write output
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': bands,
                'dtype': dtype,
                'crs': target_crs,
                'transform': target_transform,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512
            }
            
            print(f"Writing mosaic to: {mosaic_path}")
            with rasterio.open(mosaic_path, 'w', **profile) as dst:
                dst.write(output_array)
            
            print(f"Mosaic created successfully using warping method!")
            print(f"Mosaic dimensions: {width} x {height}")
            
            self.mosaic_path = str(mosaic_path)
            return str(mosaic_path)
            
        except Exception as e:
            print(f"Error creating mosaic with warping: {e}")
            raise

    def create_tiles(self, mosaic_path: str, tile_dir_name: str = "tiles") -> str:
        """
        Break the mosaic into tiles for GIS applications.
        
        Args:
            mosaic_path (str): Path to the mosaic file
            tile_dir_name (str): Name of the directory to store tiles
            
        Returns:
            str: Path to the tiles directory
        """
        print(f"\n" + "="*60)
        print("CREATING TILES")
        print("="*60)
        
        tiles_dir = self.output_directory / tile_dir_name
        tiles_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with rasterio.open(mosaic_path) as src:
                print(f"Source mosaic: {mosaic_path}")
                print(f"Dimensions: {src.width} x {src.height}")
                print(f"Tile size: {self.tile_size} x {self.tile_size}")
                
                # Calculate number of tiles
                cols = math.ceil(src.width / self.tile_size)
                rows = math.ceil(src.height / self.tile_size)
                total_tiles = cols * rows
                
                print(f"Will create {cols} x {rows} = {total_tiles} tiles")
                
                tile_count = 0
                
                for row in range(rows):
                    for col in range(cols):
                        # Calculate window
                        col_start = col * self.tile_size
                        row_start = row * self.tile_size
                        col_end = min(col_start + self.tile_size, src.width)
                        row_end = min(row_start + self.tile_size, src.height)
                        
                        # Create window
                        window = windows.Window(
                            col_off=col_start,
                            row_off=row_start,
                            width=col_end - col_start,
                            height=row_end - row_start
                        )
                        
                        # Read data for this tile
                        tile_data = src.read(window=window)
                        
                        # Skip empty tiles (all zeros or no data)
                        if np.all(tile_data == 0) or np.all(np.isnan(tile_data)):
                            continue
                        
                        # Calculate transform for this tile
                        tile_transform = windows.transform(window, src.transform)
                        
                        # Create tile metadata
                        tile_meta = src.meta.copy()
                        tile_meta.update({
                            'width': window.width,
                            'height': window.height,
                            'transform': tile_transform,
                            'compress': 'lzw'
                        })
                        
                        # Create tile filename
                        tile_filename = f"tile_{row:04d}_{col:04d}.tif"
                        tile_path = tiles_dir / tile_filename
                        
                        # Write tile
                        with rasterio.open(tile_path, 'w', **tile_meta) as dest:
                            dest.write(tile_data)
                        
                        tile_count += 1
                        
                        if tile_count % 50 == 0:
                            print(f"Created {tile_count} tiles...")
                
                print(f"Successfully created {tile_count} tiles in {tiles_dir}")
                
                # Create tile index file for GIS applications
                self.create_tile_index(tiles_dir)
                
                return str(tiles_dir)
                
        except Exception as e:
            print(f"Error creating tiles: {e}")
            raise
    
    def create_tile_index(self, tiles_dir: Path) -> None:
        """
        Create a tile index shapefile for GIS applications.
        
        Args:
            tiles_dir (Path): Directory containing tiles
        """
        try:
            import geopandas as gpd
            from shapely.geometry import box
            
            print("Creating tile index shapefile...")
            
            # Get all tile files
            tile_files = list(tiles_dir.glob("tile_*.tif"))
            
            if not tile_files:
                print("No tile files found for indexing")
                return
            
            # Create index data
            index_data = []
            
            for tile_file in tile_files:
                try:
                    with rasterio.open(tile_file) as src:
                        bounds = src.bounds
                        geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                        
                        index_data.append({
                            'geometry': geometry,
                            'filename': tile_file.name,
                            'path': str(tile_file)
                        })
                except Exception as e:
                    print(f"Warning: Could not index {tile_file}: {e}")
            
            if index_data:
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(index_data)
                
                # Set CRS from first tile
                with rasterio.open(tile_files[0]) as src:
                    gdf.crs = src.crs
                
                # Save shapefile
                index_path = tiles_dir / "tile_index.shp"
                gdf.to_file(index_path)
                print(f"Tile index created: {index_path}")
            
        except ImportError:
            print("GeoPandas not available. Skipping tile index creation.")
            print("To create tile index, install with: pip install geopandas")
        except Exception as e:
            print(f"Error creating tile index: {e}")
    
    def create_overview_image(self, mosaic_path: str, overview_filename: str = "mosaic_overview.png") -> None:
        """
        Create an overview image of the mosaic for visualization.
        
        Args:
            mosaic_path (str): Path to the mosaic file
            overview_filename (str): Name of the overview image file
        """
        print(f"\n" + "="*60)
        print("CREATING OVERVIEW IMAGE")
        print("="*60)
        
        try:
            with rasterio.open(mosaic_path) as src:
                # Read data with downsampling for overview
                overview_factor = max(1, min(src.width, src.height) // 2000)
                
                if overview_factor > 1:
                    overview_data = src.read(
                        out_shape=(src.count, src.height // overview_factor, src.width // overview_factor),
                        resampling=Resampling.bilinear
                    )
                else:
                    overview_data = src.read()
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 12))
                
                if src.count == 1:
                    # Single band
                    im = ax.imshow(overview_data[0], cmap='viridis', aspect='equal')
                    plt.colorbar(im, ax=ax)
                elif src.count >= 3:
                    # RGB or multispectral
                    rgb_data = np.transpose(overview_data[:3], (1, 2, 0))
                    # Normalize to 0-1 range
                    rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())
                    ax.imshow(rgb_data, aspect='equal')
                
                ax.set_title("Mosaic Overview")
                ax.set_xlabel("Pixels (X)")
                ax.set_ylabel("Pixels (Y)")
                
                # Save overview
                overview_path = self.output_directory / overview_filename
                plt.savefig(overview_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Overview image saved: {overview_path}")
                
        except Exception as e:
            print(f"Error creating overview image: {e}")
    
    def process_directory(self, create_tiles: bool = True, create_overview: bool = True) -> dict:
        """
        Complete processing pipeline: find files, create mosaic, and generate tiles.
        
        Args:
            create_tiles (bool): Whether to create tiles
            create_overview (bool): Whether to create overview image
            
        Returns:
            dict: Dictionary with paths to created files
        """
        print("="*80)
        print("SAS TIFF MOSAIC PROCESSOR")
        print("="*80)
        print(f"Input directory: {self.input_directory}")
        print(f"Output directory: {self.output_directory}")
        
        results = {
            'mosaic_path': None,
            'tiles_dir': None,
            'overview_path': None,
            'tiff_files_count': 0
        }
        
        try:
            # Step 1: Find TIFF files
            tiff_files = self.find_tiff_files()
            if not tiff_files:
                raise ValueError("No TIFF files found in the input directory")
            
            results['tiff_files_count'] = len(tiff_files)
            
            # Step 2: Inspect files
            self.inspect_tiff_files(tiff_files)
            
            # Step 3: Create mosaic (try standard method first, then warping method)
            try:
                mosaic_path = self.create_mosaic(tiff_files)
            except Exception as e:
                print(f"Standard mosaic creation failed: {e}")
                print("Attempting alternative method with warping...")
                mosaic_path = self.create_mosaic_with_warping(tiff_files)
            
            results['mosaic_path'] = mosaic_path
            
            # Step 4: Create tiles (optional)
            if create_tiles:
                tiles_dir = self.create_tiles(mosaic_path)
                results['tiles_dir'] = tiles_dir
            
            # Step 5: Create overview image (optional)
            if create_overview:
                overview_filename = "mosaic_overview.png"
                self.create_overview_image(mosaic_path, overview_filename)
                results['overview_path'] = str(self.output_directory / overview_filename)
            
            print(f"\n" + "="*60)
            print("PROCESSING COMPLETE!")
            print("="*60)
            print(f"Processed {results['tiff_files_count']} TIFF files")
            print(f"Mosaic: {results['mosaic_path']}")
            if results['tiles_dir']:
                print(f"Tiles: {results['tiles_dir']}")
            if results['overview_path']:
                print(f"Overview: {results['overview_path']}")
            
            return results
            
        except Exception as e:
            print(f"Error in processing: {e}")
            raise


def main():
    """
    Main function to run the mosaic processing pipeline.
    """
    # Configuration
    INPUT_DIRECTORY = r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\crop"
    OUTPUT_DIRECTORY = r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\mosaic_output"
    TILE_SIZE = 1024  # Size of tiles in pixels
    
    # Create processor
    processor = TiffMosaicProcessor(
        input_directory=INPUT_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        tile_size=TILE_SIZE
    )
    
    # Process the directory
    try:
        results = processor.process_directory(
            create_tiles=True,
            create_overview=True
        )
        
        print("\n" + "="*60)
        print("USAGE INSTRUCTIONS")
        print("="*60)
        print("1. The mosaic file can be opened directly in GIS applications like QGIS or ArcGIS")
        print("2. Individual tiles can be loaded for better performance with large datasets")
        print("3. Use the tile_index.shp file to load all tiles as a catalog in GIS")
        print("4. The overview image provides a quick visual reference")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import rasterio
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print("Required packages not found. Please install with:")
        print("pip install rasterio matplotlib numpy")
        print("Optional: pip install geopandas (for tile index creation)")
        exit(1)
    
    main()