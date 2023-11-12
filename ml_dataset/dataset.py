import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT, DenseVector
from pyspark.sql.functions import *
import json 
import random
import numpy as np

class mlDataset(Dataset):
    def __init__(self, dataset_config, partition="train"):
        spark = SparkSession.builder.appName("YourAppName").getOrCreate()

        df = spark.read.parquet(dataset_config["data_path"]).select("browserFamily", "deviceType", "os", "country").filter("deviceType != ''").na.drop()
        self.categorical_columns = ["browserFamily","deviceType", "os","country"]

        # Initialize empty lists for StringIndexer and OneHotEncoder stages
        indexers = []
        encoders = []

        # Iterate through each categorical column and add StringIndexer and OneHotEncoder stages to the lists
        for col in self.categorical_columns:
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
        onehot_encoded_columns = [f"{col}_onehot" for col in self.categorical_columns]
        self.df_encoded = df_encoded.select(onehot_encoded_columns)

        self.df_encoded_pd = self.df_encoded.toPandas()

        # mask index
        self.mask_index = -1
        self.vocab_sizes = self.__getVocabSizes__()

        if partition == "train":
            self.df_encoded_pd = self.df_encoded_pd.iloc[1000:]
        else:
            self.df_encoded_pd = self.df_encoded_pd.iloc[:1000]
    def __len__(self):
        return len(self.df_encoded_pd)
    
    def __getVocabSizes__(self):
        sizes = {}
        for col_name in  self.categorical_columns:
            sizes[col_name] = (len(self.df_encoded.schema[f"{col_name}_onehot"].metadata["ml_attr"]["attrs"]["binary"]))
        return sizes 
    
    def __indexToString__(self, indices):
        output = []
        assert len(indices) == len(self.categorical_columns), "input length != # of categories"

        for i, col_name in enumerate(self.categorical_columns):
            if indices[i] == -1:
                output.append(-1)
            else:
                idx_to_str = self.df_encoded.schema[f"{col_name}_onehot"].metadata["ml_attr"]["attrs"]["binary"]

                this_str = idx_to_str[indices[i]]
                assert this_str["idx"] == indices[i]
                output.append(this_str["name"])

        return output
    
    def __oneHotToString__(self, oneHots):
        indices = []
        for onehot in oneHots:
            if type(onehot) == int and onehot == -1:
                indices.append(-1)
            else:
                indices.append(np.argmax(onehot))

        return self.__indexToString__(indices)

    # TODO: need to optimize
    def __stringToOnehot__(self, strings):
        """
            string: list of class labels 
        """
        
        output = []
        for i, col_name in enumerate(self.categorical_columns):
            if strings[i] == -1:
                output.append(torch.full((1, self.vocab_sizes[col_name]), self.mask_index))
            else:
                idx_to_str = self.df_encoded.schema[f"{col_name}_onehot"].metadata["ml_attr"]["attrs"]["binary"]

                this_str_index = [str_name["idx"] for str_name in idx_to_str if str_name["name"] == strings[i]]

                assert len(this_str_index)==1, "input not in vocab or multiple found in vocab"
                one_hot = np.zeros(len(idx_to_str))
                one_hot[this_str_index[0]] = 1
                
                output.append(one_hot)

        return output



    def __getitem__(self, index):
        # dataset.__getitem__(0)
        data_point = self.df_encoded_pd.iloc[index]

        input = {}
        label = {}
        masked_position = []

        has_input_flag = False
        for cols in self.categorical_columns:
            prob = random.random()
            column_label_array = torch.tensor(DenseVector(data_point[f"{cols}_onehot"].toArray()).toArray(), dtype = int)
            if prob > 0.8:
                # 0.2 percent it will not be replaced by mask
                input[cols] = column_label_array
                masked_position.append(torch.tensor(0))
                has_input_flag = True
                label[cols] =  torch.tensor([-100])
            else:
                # 0.8 percent chance it will be replaced by mask
                input[cols] = torch.full(column_label_array.shape, self.mask_index)
                masked_position.append(torch.tensor(1))
                
                label[cols] =  torch.nonzero(column_label_array)[0]


        output = {"input": input,
                  "label" : label,
                  "masked_position": torch.stack(masked_position),
                  }
        

        if not has_input_flag:
            return self.__getitem__(index)

        return output

