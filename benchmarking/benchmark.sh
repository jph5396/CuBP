#!/bin/bash
set -e

# Update these values to test other things. 
IMAGE_SIZES=(512 1024 1536)
NUM_PULSES=(1000 2500 5000)
BACKENDS=(numpy cuda)
CPHD_FILES=(
    capella_egypt.cphd
    jeddah-tower.cphd
)
LOG_FILE="benchmarking-logs.log"

for cphd_file in "${CPHD_FILES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        for size in "${IMAGE_SIZES[@]}"; do
            for pulses in "${NUM_PULSES[@]}"; do
                echo "Running cubp with the following commands: " 
                echo "cphd=${cphd_name} backend=${backend} size=${size}x${size} pulses=${pulses}"

                cubp \
                    --cphd_file "$cphd_file" \
                    --backend "$backend" \
                    --image_bounds.x "$size" \
                    --image_bounds.y "$size" \
                    --pulse_limit "$pulses" \
                    --no-save_image \
                    --image_spacing 0.5 \
                    --log_file "$LOG_FILE"
            done
        done
    done
done

echo "Done"
