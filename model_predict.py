from pyspark import SparkConf, SparkContext, SQLContext
from pyexpat import model
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import DecisionTreeClassifier

#instantiate spark session
conf = (SparkConf().setAppName("WineQuality-Prediction"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)


train_dataset = "/tmp/ValidationDataset.csv" #"s3://myprojectdataset/TrainingDataset.csv" #sys.argv[1] #"/tmp/TrainingDataset.csv"

print("Reading data..")
df_validation = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(train_dataset)

features = df_validation.columns

features = [c for c in df_validation.columns if c != 'quality'] #Drop quality column
print(features)

df_validation.select(features).describe().toPandas().transpose()


va = VectorAssembler(inputCols=features, outputCol="features")
df_validation = va.transform(df_validation)


print("===================Random Forest model===================")

rf = RandomForestClassifier(featuresCol = 'features', labelCol = features[-1] , numTrees=60, maxBins=32, maxDepth=4, seed=42)

rf_model = RandomForestClassificationModel.load("/tmp/rf-trained.model")
predictions = rf_model.transform(df_validation)

print("Saving the trained model to S3 bucket..")
rf_model.write().overwrite().save("/tmp/rf-predicted.model")

print("Evaluate the trained model...")

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
f1score = evaluator.evaluate(predictions)
print("F1-Score = %s" % (f1score))









