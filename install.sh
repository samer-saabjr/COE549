#!/bin/bash

################################################################

# Remove Python cache directories
find src/llms_course -type d -name '__pycache__' -exec rm -rf {} +
find src/llms_course -type d -name '*.egg-info' -exec rm -rf {} +

################################################################

# Install the project
# pip install -e .
