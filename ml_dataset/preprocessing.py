from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import json 

# Create a Spark session
spark = SparkSession.builder.appName("YourAppName").getOrCreate()

df = spark.read.parquet("ml-dataset/pm-dataset.parquet").select("browserFamily", "deviceType", "os", "country").filter("deviceType != ''").na.drop()


categorical_columns = ["browserFamily","deviceType", "os","country"]

# Initialize empty lists for StringIndexer and OneHotEncoder stages
indexers = []
encoders = []

# Iterate through each categorical column and add StringIndexer and OneHotEncoder stages to the lists
for col in categorical_columns:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
    encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_onehot", dropLast = False)
    indexers.append(indexer)
    encoders.append(encoder)

# Create a list of all stages to be used in the pipeline
all_stages = indexers + encoders

# Define the pipeline with all stages
pipeline = Pipeline(stages=all_stages)

# Fit and transform the data using the pipeline
model = pipeline.fit(df)
df_encoded = model.transform(df)


# save indexer and encoder 
# for i in range(len(categorical_columns)):
#     indexers[i].save(f"{categorical_columns[i]}_string_indexer")

# Select only the one-hot encoded columns
onehot_encoded_columns = [f"{col}_onehot" for col in categorical_columns]
df_encoded = df_encoded.select(onehot_encoded_columns)

# Show the resulting DataFrame with one-hot encoded columns
df_encoded.show()
import pdb
pdb.set_trace()