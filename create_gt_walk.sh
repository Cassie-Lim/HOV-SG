#!/bin/bash

# List of scene IDs
scene_ids=(
    # "00861-GLAQ4DNUx5U"
    # "00862-LT9Jq6dN3Ea"
    "00873-bxsVRursffK"
    # "00877-4ok3usBNeis"
    # "00890-6s7QHgap2fW"
)

# Command template
command_template="python hovsg/data/hm3dsem/create_hm3dsem_walks_gt.py main.scene_id="

# Run the command for each scene ID
for scene_id in "${scene_ids[@]}"; do
    command="${command_template}${scene_id}"
    echo "Executing: $command"
    $command
done
