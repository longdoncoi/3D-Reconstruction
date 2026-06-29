#!/bin/bash
set -e
git pull

# Build and Test C++
cmake --build build
ctest --output-on-failure

# Lint Python
ruff check AITraining/ --output-format=github

# Commit and push
git add .
git commit -m "Agent task completed"
git push
