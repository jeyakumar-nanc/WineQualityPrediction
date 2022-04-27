# WineQualityPrediction
# Requirement:
The purpose of this individual assignment is to learn how to develop parallel machine learning (ML) applications in Amazon AWS cloud platform. Specifically, you will learn: (1) how to use Apache Spark to train an ML model in parallel on multiple EC2 instances; (2) how to use Spark’s MLlib to develop and use an ML model in the cloud; (3) How to use Docker to create a container for your ML model to simplify model deployment
# Create EMR cluster for parallel training
1.	In the AWS dashboard under the analytics section click EMR
2.	Now Click Create Cluster
3.	In the General Configuration for Cluster Name type desired cluster name.
  a.	Under Software configuration` in the application column click the button which shows `Spark: Spark 2.4.7 on Hadoop 2.10.1 YARN and Zeppelin 0.8.2.
  b. 	Under Hardware Configuration click m4.large rather than the default m5.xlarge as the default m5.xlarge incurs a cost of $0.043/hr in contrast to the $0.03 for m4.large. Keep in mind that EMR incurs an additional 25% cost post first usage.
  c.	Select 4 instances under the column Number of instances
  d.	Under Security and access click the EC2 key pair already created else create a new one
  
  ![image](https://user-images.githubusercontent.com/66985675/165438805-4c23421c-b7d6-4d76-99ec-2ac3e888c38a.png)

 ![image](https://user-images.githubusercontent.com/66985675/165438839-ce891277-052d-44b4-81d0-0ea3dd0347e2.png)

 
•	EMR cluster created 4 ec2 instances

![image](https://user-images.githubusercontent.com/66985675/165438848-da8e06db-3a0e-4915-b642-b118ac455d9f.png)

  
# S3 bucket creation to store the dataset and model output
![image](https://user-images.githubusercontent.com/66985675/165438881-b35d1bba-9e5a-4cdd-8592-5026323b7f81.png)

 
•	Once the EMR is ready, connect to the master node using SSH via PuTTY terminal 
 ![image](https://user-images.githubusercontent.com/66985675/165438937-bebc0814-cc15-43eb-9806-adc4ac362dac.png)

 
•	Run the job by executing the following command in EMR terminal
•	My model expects two parameters  => 1. Dataset location 2.Output location to save the data
•	Using decision tree classifier and randomforest classifier for training the data
spark-submit model_train.py "s3://myprojectdataset/TrainingDataset.csv" "s3://myprojectdataset/"

•	Output of the models will be stored in S3 bucket (these will be later used in prediction)
 ![image](https://user-images.githubusercontent.com/66985675/165439090-c5a6ca90-780d-4c71-b2f5-55dcf572e9d9.png)

Accuracy of Decision tree seems to be overfitting so lets do prediction using Random forest classifier
![image](https://user-images.githubusercontent.com/66985675/165439111-f918741c-72aa-46e3-aaaa-c3eb7901b0c2.png)

 
# Create EC2 Instance for Prediction 
Ubuntu instance is launched as follows :
•	Go to EC2 dashboard and click on "Launch instances”
•	Select Ubuntu machine images
•	In Choose an Instance type select "t2.micro" and click on "Review and Launch”
•	Click on "Launch"
•	Create a New key pair or choose an existing one and click on "Launch"
![image](https://user-images.githubusercontent.com/66985675/165439190-7b82fce2-e6fc-4a89-9929-96b2ecd57b12.png)


The results of my model trained in EMR is stored in my s3 bucket. So inorder to pass it to my prediction model in Ec2, we download/sync s3 bucket data using aws cli.
Execute the following commands to install aws cli in ec2
apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

Configure AWS credentials by executing “aws configure” command
 
Run aws s3 sync to download s3 objects into ec2 which will be later used in the project
aws s3 sync s3://myprojectdataset/ model/
![image](https://user-images.githubusercontent.com/66985675/165439221-6e6ac9bb-de14-43fc-b0f3-9ac08f17517b.png)

 ![image](https://user-images.githubusercontent.com/66985675/165439238-8193b854-c0ab-41c0-8dae-109b2fd9ac4a.png)

 
# Spark Installation
•	SSH into the EC2 via MobaXterm or putty and execute the following script to install the dependencies for Spark and Hadoop
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

•	After the setup is complete, set up the environment variables for pyspark in ~/.bashrc
export SPARK_HOME=/home/ubuntu/spark-3.0.0-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_PYTHON=python3
export PATH=$PATH:$JAVA_HOME/jre/bin

•	Reboot the server to make the changes apply.
•	To verify spark installation, type in pyspark command in the shell terminal. On successful, you should be able to something like below.
![image](https://user-images.githubusercontent.com/66985675/165439291-e3970907-e8a0-43c5-b079-0545f20f0d24.png)
 
# Model Prediction Without Docker
spark-submit model_train.py model/TrainingDataset.csv .
spark-submit model_predict.py model/ValidationDataset.csv .

 

Model_train  

Model prediction
I have used decision tree classifier and random forest classifier for training, based on the accuracy from these two classifiers in the train dataset decision tree classifier seems to overfit the data. So, for prediction I’m using random forest classifier for model prediction, which gave the accuracy of 95.6% percent, error – 4.3% and F1 score – 93.6
![image](https://user-images.githubusercontent.com/66985675/165439385-a584328e-f76b-47dd-b473-3bbe466a4c4f.png)


# Docker Installation
•	Executing the following script in the ec2 instance to install docker
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

For spark environment setup, I have pulled the spark+Hadoop+java package from the existing docker image and added my model_predict.py, Validation dataset and trained model. So Dockerfile contains following commands to build my wine_quality_app image. (Since the spark image is more 1.5GB, I had expanded the volume from t2 micro to t3 medium and resized the boot memory using the following commands and reboot the instance
Sudo resize2fs /dev/xvda1

•	Docker file creation for wine_quality_app image
FROM datamechanics/spark:3.0-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3
RUN conda install -y numpy
#RUN aws s3 sync s3://myprojectdataset/ model/

WORKDIR /opt/wine-quality-app

COPY model_predict.py .
ADD model/ValidationDataset.csv .
ADD model ./model/



•	Execute the following command to build wine_quality_app. This should be executed from the same folder where you have the Dockerfile created
docker build -t wine-quality-app .

![image](https://user-images.githubusercontent.com/66985675/165439442-4689e91a-b548-4cfe-8338-6f84d18316eb.png)
 
•	Execute docker image ls to cross check the docker image
![image](https://user-images.githubusercontent.com/66985675/165439484-7f72b409-e18a-4a2d-a6f9-300e076e52f7.png)

# Model Prediction with Docker
•	Execute the command to create container and run the model prediction using spark
docker run wine-quality-app driver model_predict.py model/ValidationDataset.csv model/

![image](https://user-images.githubusercontent.com/66985675/165439504-e002a50d-f8be-4af6-9186-4746cb51624f.png)

  
•	I have used decision tree classifier and random forest classifier for training, based on the accuracy from these two classifiers in the train dataset decision tree classifier seems to overfit the data. So, for prediction I’m using random forest classifier for model prediction, which gave the accuracy of 95.6% percent, error – 4.3% and F1 score – 93.6

![image](https://user-images.githubusercontent.com/66985675/165439578-2afcff2c-75df-4967-9509-e7f702516882.png)

# Docker Hub set up and execution from docker repo
1.	Tag the built image with the repo name
 docker tag wine-quality-app:latest jeyakumarn/wine-quality:latest
2.	Login to docker using your credentials
3.	Push the image to docker repo
 docker push jeyakumarn/wine-quality:latest

Docker image ls  
4.	Run the command to pull image from your repo
docker run jeyakumarn/wine-quality driver model_predict.py model/ValidationDataset.csv model/

# Repository Links
Docker Repo: https://hub.docker.com/repository/docker/jeyakumarn/wine-quality
Github Repo: https://github.com/jeyakumar-nanc/WineQualityPrediction
