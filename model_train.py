from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType, DoubleType

#instantiate spark session
spark = SparkSession.builder.appName("WineQuality-Training").getOrCreate()

train_dataset = "/tmp/TrainingDataset.csv" #"s3://myprojectdataset/TrainingDataset.csv" #sys.argv[1] #"/tmp/TrainingDataset.csv"

print("Reading data..")
df_training = spark.read.format("csv").load(train_dataset, header=True, sep=";")

df_training = df_training.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")

#df_training.show()

pd.DataFrame(df_training.take(10), columns=df_training.columns).transpose()

df_training = df_training \
        .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
        .withColumn("volatile_acidity", col("volatile_acidity").cast(DoubleType())) \
        .withColumn("citric_acid", col("citric_acid").cast(DoubleType())) \
        .withColumn("residual_sugar", col("residual_sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free_sulfur_dioxide", col("free_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("total_sulfur_dioxide", col("total_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("label", col("label").cast(IntegerType()))

features = df_training.columns
features = features[:-1]

df_training.select(features).describe().toPandas().transpose()


va = VectorAssembler(inputCols=features, outputCol="features")
df_training = va.transform(df_training)

print("Training random forest model..")

features = df_training.columns

#print(features)

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rf_model = rf.fit(df_training)
predictions = rf_model.transform(df_training)

print("Saving the trained model to S3 bucket..")
rf_model.save("s3://winequalityproject/trained-model")

print("Evaluate the trained model...")

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1score = evaluator.evaluate(predictions)
print("F1-Score = %s" % (f1score))