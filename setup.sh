#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Train models (generate .pkl files)
python models/compare_multitarget_models.py
