import sys
from pyspark import SparkConf, SparkContext, SQLContext
from pyexpat import model
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.classification import DecisionTreeClassifier


if(len(sys.argv)==1):
    raise Exception("Enter valid parameters to submit the job. Execute the job by providing proper parameters => model_train.py <dataset location> <output path>")


#instantiate spark session
conf = (SparkConf().setAppName("WineQuality-Training"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

dt_trained_model_result = sys.argv[2]+"dt-trained.model"  
rf_trained_model_result = sys.argv[2]+"rf-trained.model"  
train_dataset = sys.argv[1]

print(f"Reading data from {train_dataset} ..")
df_training = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(train_dataset)

features = df_training.columns

#features = [c for c in df_training.columns if c != 'quality'] #Drop quality column
#print(features)

#df_training.select(features).describe().toPandas().transpose()


va = VectorAssembler(inputCols=features, outputCol="features")
df_training = va.transform(df_training)

print("===================Decision Tree Classifier model===================")
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = features[-1], maxDepth =2)
dt_Model = dt.fit(df_training)
dt_predictions = dt_Model.transform(df_training)

print(f"Saving the trained model to {dt_trained_model_result} ..")
dt_Model.write().overwrite().save(dt_trained_model_result)

print("Evaluate the trained model...")

dt_evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
dt_accuracy = dt_evaluator.evaluate(dt_predictions)
print("Accuracy = %s" % (dt_accuracy))
print("Test Error = %s" % (1.0 - dt_accuracy))

dt_evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
dt_f1score = dt_evaluator.evaluate(dt_predictions)
print("F1-Score = %s" % (dt_f1score))



print("===================Random Forest model===================")

#print(features)

rf = RandomForestClassifier(featuresCol = 'features', labelCol = features[-1] , numTrees=60, maxBins=32, maxDepth=4, seed=42)

rf_model = rf.fit(df_training)
predictions = rf_model.transform(df_training)

print(f"Saving the trained model to {rf_trained_model_result} ..")
rf_model.write().overwrite().save(rf_trained_model_result)

print("Evaluate the trained model...")

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
f1score = evaluator.evaluate(predictions)
print("F1-Score = %s" % (f1score))

#print(rf_model.featureImportances)


