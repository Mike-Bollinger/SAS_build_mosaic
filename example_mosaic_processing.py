"""
Simple example of how to use the SAS_merge_into_mosaic.py script

This example shows how to process TIFF files from your specific directory.
"""

from pathlib import Path
import sys
import os

# Add the current directory to Python path so we can import the mosaic processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SAS_merge_into_mosaic import TiffMosaicProcessor


def example_processing():
    """
    Example of processing TIFF files into a mosaic and tiles.
    """
    
    # Your specific directories
    input_dir = r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\crop"
    output_dir = r"E:\EN2501\SAS\DIVE049_SN401\processing\geotiff\EN2501_DIVE049_10cm_1_13_67\mosaic_output"
    
    # Check if input directory exists
    if not Path(input_dir).exists():
        print(f"Input directory does not exist: {input_dir}")
        print("Please update the input_dir variable with the correct path.")
        return
    
    # Create the mosaic processor
    processor = TiffMosaicProcessor(
        input_directory=input_dir,
        output_directory=output_dir,
        tile_size=1024  # 1024x1024 pixel tiles
    )
    
    # Process all TIFF files in the directory
    try:
        results = processor.process_directory(
            create_tiles=True,      # Create tile packages
            create_overview=True    # Create overview image
        )
        
        print("\n" + "="*50)
        print("PROCESSING RESULTS:")
        print("="*50)
        print(f"✓ Processed {results['tiff_files_count']} TIFF files")
        print(f"✓ Created mosaic: {results['mosaic_path']}")
        if results['tiles_dir']:
            print(f"✓ Created tiles in: {results['tiles_dir']}")
        if results['overview_path']:
            print(f"✓ Created overview: {results['overview_path']}")
        
        print(f"\n" + "="*50)
        print("NEXT STEPS:")
        print("="*50)
        print("1. Open the mosaic file in QGIS or ArcGIS:")
        print(f"   {results['mosaic_path']}")
        print("\n2. For large datasets, use individual tiles for better performance")
        print(f"   Load tiles from: {results['tiles_dir']}")
        print("\n3. Use the tile index shapefile to load all tiles at once:")
        print(f"   {Path(results['tiles_dir']) / 'tile_index.shp'}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Check that:")
        print("1. The input directory contains TIFF files")
        print("2. You have write permissions to the output directory")
        print("3. All required Python packages are installed")


if __name__ == "__main__":
    example_processing()
