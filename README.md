### Cloud Optimized Geotiff to H3 Cell

This script processes Cloud Optimized GeoTIFFs (COGs) and converts them to H3 hexagonal grids, storing the results in a PostgreSQL database. It supports multi-band rasters, resampling, and various processing options.

## Features

- Download and process COGs from URLs
- Convert raster data to H3 hexagons
- Support for multi-band rasters
- Customizable H3 resolution
- Various resampling methods
- Option to preserve raster value range during resampling
- Efficient database storage using PostgreSQL

## Prerequisites

- Python 3.7+
- PostgreSQL database

- Ensure your input COG is in WGS 1984 projection. If not, reproject it using gdalwarp

Example : 
```shell
gdalwarp -overwrite input.tif output.tif -s_srs EPSG:32645 -t_srs EPSG:4326
```

- Set nodata values to 0 using `gdalwarp`, 
```shell
gdalwarp -dstnodata 0 input.tif output.tif
```

- Convert a GeoTIFF to COG format:
```shell
gdal_translate -of COG input.tif output_cog.tif
```


## Environment Variables

Set the following environment variables or use default values:

- `DATABASE_URL`: PostgreSQL connection string (default: "postgresql://postgres:postgres@localhost:5432/postgres")
- `STATIC_DIR`: Directory to store downloaded COGs (default: "static")

Example:
```shell
export DATABASE_URL="postgresql://user:password@host:port/database"
export STATIC_DIR="/path/to/cog/storage"
```
## Usage

Run the script with the following command:
```shell
python cog2h3.py --cog <COG_URL> --table <TABLE_NAME> [OPTIONS]
```
### Required Arguments:

- `--cog`: URL of the Cloud Optimized GeoTIFF (must be in WGS84 projection)
- `--table`: Name of the database table to store results

### Optional Arguments:

- `--res`: H3 resolution level (default: 8)
- `--preserve_range`: Preserve the value range of the raster while resampling
- `--multiband`: Process all bands of a multi-band raster
- `--sample_by`: Resampling method (default: "bilinear")

Available resampling methods: nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, gauss, max, min, med, q1, q3, sum, rms

### Example:
```shell
python cog2h3.py --cog my-cog.tif --table cog_h3 --res 8
```

## Contribute 

Contributions are Welcome ! स्वागतम

