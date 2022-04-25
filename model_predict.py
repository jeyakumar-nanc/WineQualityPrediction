import sys
from pyspark import SparkConf, SparkContext, SQLContext
from pyexpat import model
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import DecisionTreeClassifier


if(len(sys.argv)==1):
    raise Exception("Enter valid parameters to submit the job. Execute the job by providing proper parameters => model_predict.py <dataset location> <output path>")
    
#instantiate spark session
conf = (SparkConf().setAppName("WineQuality-Prediction"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

dt_trained_model_result = sys.argv[2]+"dt-trained.model"  
rf_trained_model_result = sys.argv[2]+"rf-trained.model"  
rf_predicted_model_result = sys.argv[2]+"rf-predicted.model"
train_dataset = sys.argv[1] 

print(f"Reading data from {train_dataset}..")
df_validation = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(train_dataset)

features = df_validation.columns

features = [c for c in df_validation.columns if c != 'quality'] #Drop quality column


df_validation.select(features).describe().toPandas().transpose()


va = VectorAssembler(inputCols=features, outputCol="features")
df_validation = va.transform(df_validation)


print("===================Random Forest model===================")

rf = RandomForestClassifier(featuresCol = 'features', labelCol = features[-1] , numTrees=60, maxBins=32, maxDepth=4, seed=42)

rf_model = RandomForestClassificationModel.load(rf_trained_model_result)

predictions = rf_model.transform(df_validation)

print(f"Saving the trained model to {rf_predicted_model_result} ..")
rf_model.write().overwrite().save(rf_predicted_model_result) 

print("Evaluate the trained model...")

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
f1score = evaluator.evaluate(predictions)
print("F1-Score = %s" % (f1score))


