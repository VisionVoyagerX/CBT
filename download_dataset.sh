#!/bin/bash

# Function to get the direct download link from Google Drive
get_direct_download_link() {
    local file_id="$1"
    local confirm=$(curl -s -c /tmp/gdcookie "https://drive.google.com/uc?export=download&id=${file_id}" | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    curl -s -b /tmp/gdcookie "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" -L -o "${2}"
    rm -f /tmp/gdcookie
}

# Download dataset function
download_dataset() {
    local dataset_name="$1"
    local file_id="$2"
    local file_name="$3"
    local base_dir="$4"

    echo "============================${dataset_name}================================"
    echo "Downloading ${dataset_name} dataset"

    # Create directory structure if it doesn't exist
    mkdir -p "$base_dir"

    # Get the direct download link and download the file
    get_direct_download_link "$file_id" "${base_dir}/${file_name}"

    echo "FINISHED DOWNLOADING ${dataset_name} DATASET SUCCESSFULLY."
}

# Download GF2 dataset
download_dataset "GF2" "14Kc3kvjWFUBbQ3KhoqwR7ksHg5vYVt" "test_gf2_multiExm1.h5" "dataset/GF2/test"

# Download WV3 dataset
download_dataset "WV3" "1r2pEu0_OICdZA7Dga0b2uRvJMK28JlO3" "test_wv3_multiExm1.h5" "dataset/WV3/test"

