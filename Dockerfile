FROM datamechanics/spark:3.0-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3
RUN conda install -y numpy

WORKDIR /opt/wine-quality-app

COPY model_predict.py .
ADD model/ValidationDataset.csv .
ADD model ./model/
