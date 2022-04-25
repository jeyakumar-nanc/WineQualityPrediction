#!/bin/sh
sudo apt -y update
echo "Install pip..."
sudo apt install python3-pip

echo "Install java.."
sudo apt-get install default-jre

echo "Install Py4j.."
pip install py4j

echo "Install Spark and Hadoop.."
wget http://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz
sudo tar -zxvf spark-3.0.0-bin-hadoop2.7.tgz

sudo pip install findspark

echo "Set environmental variables for Spark.."
mv spark-3.0.0-bin-hadoop2.7 /home/ubuntu/
