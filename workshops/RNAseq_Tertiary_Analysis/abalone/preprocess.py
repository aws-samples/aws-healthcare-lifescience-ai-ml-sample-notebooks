# """Feature engineers the abalone dataset."""
# import argparse
# import logging
# import os
# import pathlib
# import requests
# import tempfile

# import boto3
# import numpy as np
# import pandas as pd

# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


# # Since we get a headerless CSV file we specify the column names here.
# feature_columns_names = [
#     "sex",
#     "length",
#     "diameter",
#     "height",
#     "whole_weight",
#     "shucked_weight",
#     "viscera_weight",
#     "shell_weight",
# ]
# label_column = "rings"

# feature_columns_dtype = {
#     "sex": str,
#     "length": np.float64,
#     "diameter": np.float64,
#     "height": np.float64,
#     "whole_weight": np.float64,
#     "shucked_weight": np.float64,
#     "viscera_weight": np.float64,
#     "shell_weight": np.float64,
# }
# label_column_dtype = {"rings": np.float64}


# def merge_two_dicts(x, y):
#     """Merges two dicts, returning a new copy."""
#     z = x.copy()
#     z.update(y)
#     return z


# if __name__ == "__main__":
#     logger.debug("Starting preprocessing.")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input-data", type=str, required=True)
#     args = parser.parse_args()

#     base_dir = "/opt/ml/processing"
#     pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
#     input_data = args.input_data
#     bucket = input_data.split("/")[2]
#     key = "/".join(input_data.split("/")[3:])

#     logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
#     fn = f"{base_dir}/data/abalone-dataset.csv"
#     s3 = boto3.resource("s3")
#     s3.Bucket(bucket).download_file(key, fn)

#     logger.debug("Reading downloaded data.")
#     df = pd.read_csv(
#         fn,
#         header=None,
#         names=feature_columns_names + [label_column],
#         dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
#     )
#     os.unlink(fn)

#     logger.debug("Defining transformers.")
#     numeric_features = list(feature_columns_names)
#     numeric_features.remove("sex")
#     numeric_transformer = Pipeline(
#         steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
#     )

#     categorical_features = ["sex"]
#     categorical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )

#     preprocess = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features),
#         ]
#     )

#     logger.info("Applying transforms.")
#     y = df.pop("rings")
#     X_pre = preprocess.fit_transform(df)
#     y_pre = y.to_numpy().reshape(len(y), 1)

#     X = np.concatenate((y_pre, X_pre), axis=1)

#     logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
#     np.random.shuffle(X)
#     train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

#     logger.info("Writing out datasets to %s.", base_dir)
#     pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
#     pd.DataFrame(validation).to_csv(
#         f"{base_dir}/validation/validation.csv", header=False, index=False
#     )
#     pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_test_split_ratio", type=float, default=0.2)
    parser.add_argument("--gene_count", type=int, default=10000)
    parser.add_argument("--local_path", type=str, default="/opt/ml/processing")
    parser.add_argument("--hiseq_url", type=str, default="https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2_PANCAN.gz")
    parser.add_argument("--brca_clinical_matrix_url", type=str, default="https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix")
    parser.add_argument("--create_test_data", default=False, action="store_true")

    return parser.parse_known_args()


if __name__ == "__main__":

    ### Command line parser
    args, _ = _parse_args()

    DATA_DIR = os.path.join(args.local_path, "input")
    print(f"Data directory is {DATA_DIR}")
        
    # Get TCGA BRCA Gene Expression Data
    os.system(f"wget {args.hiseq_url} -nc -nv -P {DATA_DIR}/")
    os.system(f"gzip -df {DATA_DIR}/HiSeqV2_PANCAN.gz")

    # Get TCGA BRCA Phenotype Data
    os.system(f"wget {args.brca_clinical_matrix_url} -nc -nv -P {DATA_DIR}/")

    ### Load Gene Expression RNA-seq
    print(os.listdir(DATA_DIR))
    print(os.path.join(DATA_DIR, "HiSeqV2_PANCAN"))
    genom = pd.read_csv(os.path.join(DATA_DIR, "HiSeqV2_PANCAN"), sep="\t")
    genom = genom[:int(args.gene_count)]
    genom_identifiers = genom["sample"].values.tolist()

    ### Load Phenotypes
    phenotypes = pd.read_csv(os.path.join(DATA_DIR, "BRCA_clinicalMatrix"), sep="\t")

    #### Keep `HER2_Final_Status_nature2012` target variables
    phenotypes_subset = phenotypes[
        ["sampleID", "HER2_Final_Status_nature2012"]
    ].reset_index(drop=True)
    phenotypes_subset.fillna("Negative", inplace=True)

    ### Transpose Methylation and Gene Expression datasets in order to join with Phenotypes on sampleID
    genom_transpose = (
        genom.set_index("sample")
        .transpose()
        .reset_index()
        .rename(columns={"index": "sampleID"})
    )

    ### Merge datasets
    df = pd.merge(phenotypes_subset, genom_transpose, on="sampleID", how="left")

    ### Encode target
    df["target"] = [
        0 if t == "Negative" else 1 for t in df["HER2_Final_Status_nature2012"]
    ]
    df = df.drop(["HER2_Final_Status_nature2012", "sampleID"], axis=1)
    ## Move target to first column
    df.insert(loc=0, column="target", value=df.pop("target"))
    ## Drop rows with NaN values
    df = df.dropna()

    ### Train-Valid-Test split
    # Hold out 20% of the data for validation
    train_df, val_df = train_test_split(df, test_size=float(args.train_test_split_ratio))

    print(
        f"The training data has {train_df.shape[0]} records and {train_df.shape[1]} columns."
    )
    print(
        f"The validation data has {val_df.shape[0]} records and {val_df.shape[1]} columns."
    )
    
    if args.create_test_data:
        # Split val_df into val and test sets
        test_df, val_df = train_test_split(val_df, test_size=0.5)
        print(
            f"The test data has {test_df.shape[0]} records and {test_df.shape[1]} columns."
        )  
    
    # Save data
    os.makedirs(os.path.join(args.local_path, "train"), exist_ok=True)
    training_output_path = os.path.join(args.local_path, "train/train.csv")
    train_df.to_csv(training_output_path, header=False, index=False)
    print(f"Training data saved to {training_output_path}")

    os.makedirs(os.path.join(args.local_path, "validation"), exist_ok=True)
    val_output_path = os.path.join(args.local_path, "validation/validation.csv")
    val_df.to_csv(val_output_path, header=False, index=False)
    print(f"Validation data saved to {val_output_path}")

    if args.create_test_data:   
        os.makedirs(os.path.join(args.local_path, "test"), exist_ok=True)
        test_output_path = os.path.join(args.local_path, "test/test.csv")
        test_df.to_csv(test_output_path, header=False, index=False)
        print(f"Test data saved to {test_output_path}")