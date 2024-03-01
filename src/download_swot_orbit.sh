#!/bin/bash

# Base directory to save the downloaded files
BASE_SAVE_DIR="../data/"

mkdir -p "$BASE_SAVE_DIR"

# Declare an array of URLs to be downloaded
URLS=(
    "https://www.aviso.altimetry.fr/fileadmin/documents/missions/Swot/sph_science_nadir.zip"
    "https://www.aviso.altimetry.fr/fileadmin/documents/missions/Swot/sph_science_swath.zip"
)

# Loop through each URL
for url in "${URLS[@]}"; do
    # Extract filename from URL
    filename=$(basename "$url")

    # Complete path to save the downloaded file
    save_path="$BASE_SAVE_DIR$filename"

    # Check if the file already exists
    if [[ -f "$save_path" ]]; then
        echo "The file $filename already exists at $save_path. Skipping download."
    else
        # Download the file using wget
        wget -O "$save_path" "$url"
        if [ $? -eq 0 ]; then
            echo "File $filename downloaded and saved to $save_path"
        else
            echo "Error downloading $filename"
        fi
    fi
done
