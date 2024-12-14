#!/bin/bash

# List of scene IDs
scene_ids=(
    # "00847-bCPU9suPUw9"
    # "00849-a8BtkwhxdRV"
    # "00861-GLAQ4DNUx5U"
    # "00862-LT9Jq6dN3Ea"
    "00873-bxsVRusrffk"
    # "00877-4ok3usBNeis"
    # "00890-6s7QHgap2fW"
)

# Command template
command_template="python application/create_graph.py main.dataset=hm3dsem main.dataset_path=hovsg/data/hm3dsem_walks/ main.save_path=hovsg/data/scene_graphs/ main.scene_id="

# Run the command for each scene ID
for scene_id in "${scene_ids[@]}"; do
    command="${command_template}${scene_id}"
    echo "Executing: $command"
    $command
done