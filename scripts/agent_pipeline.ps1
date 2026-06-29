git pull

# Build project (C++)
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { 
    Write-Error "C++ Build failed"
    exit 1 
}

# Run tests (C++)
ctest --output-on-failure
if ($LASTEXITCODE -ne 0) { 
    Write-Error "C++ Tests failed"
    exit 1 
}

# Lint Python code
ruff check AITraining/ --output-format=github
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python Lint failed"
    exit 1
}

# Commit and push
git add .
git commit -m "Agent task completed"
git push
