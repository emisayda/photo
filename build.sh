#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading U^2-Net model from Google Drive..."
mkdir -p saved_models
gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O saved_models/u2net.pth
