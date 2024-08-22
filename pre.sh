#!/bin/bash

set -e

for tool in gdalwarp gdal_translate; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is not installed."
        exit 1
    fi
done

input_file="$1"
basename=$(basename "$input_file" .tif)
output_file="${basename}_preprocessed.tif"

if [ -z "$input_file" ]; then
    echo "Usage: $0 <input_file.tif>"
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi

target_srs="EPSG:4326"

if [[ "$current_srs" != *"$target_srs"* ]]; then
    echo "Reprojecting $input_file to $target_srs..."
    gdalwarp -overwrite "$input_file" "${basename}_reprojected.tif" -t_srs "$target_srs"
    input_file="${basename}_reprojected.tif"
fi

echo "Setting nodata values to 0..."
gdalwarp -overwrite -dstnodata 0 "$input_file" "${basename}_nodata.tif"
rm -rf $input_file

input_file="${basename}_nodata.tif"

echo "Converting to COG format..."
gdal_translate -of COG "$input_file" "$output_file"

rm -rf $input_file

echo "Processing complete. Output file: $output_file"
