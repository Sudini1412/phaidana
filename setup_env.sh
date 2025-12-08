#!/bin/bash

# 1. Create the virtual environment if it doesn't exist
if [ ! -d "phaidana_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv phaidana_venv
fi

# 2. Activate the virtual environment
source phaidana_venv/bin/activate

# 3. Upgrade pip (good practice)
pip3 install --upgrade pip

# 4. Install dependencies from requirements.txt
echo "Installing dependencies..."
pip3 install -r requirements.txt

# 5. Install the project itself in 'editable' mode
# This replaces the hardcoded 'phaidana' path you had earlier
pip3 install -e .

echo "Setup complete! To activate the environment, run: source phaidana_venv/bin/activate"

