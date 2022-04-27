#!/bin/sh
echo "Installing docker .."
#sudo apt-get update
#sudo apt install -y docker.io
#sudo systemctl enable docker.service
#sudo systemctl status docker.service

echo "Verify docker.."
sudo docker --version

echo "Granting god permissions to ubuntu user to run docker.."
sudo usermod -a -G docker ubuntu
