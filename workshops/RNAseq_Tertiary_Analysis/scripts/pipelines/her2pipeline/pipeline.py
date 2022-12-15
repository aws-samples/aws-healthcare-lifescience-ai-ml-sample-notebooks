"""Example workflow pipeline script for brca-her2 pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CacheConfig,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.debugger import Rule, rule_configs



BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="Her2PackageGroup",
    pipeline_name="Her2Pipeline",
    base_job_prefix="Her2",
):
    """Gets a SageMaker ML Pipeline instance working with on her2 data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    ####################################################
    # Define Parameters for pipeline execution
    ####################################################    
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

    
    processing_hiseq_uri = ParameterString(name="HiSeqDataURL", default_value="https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2_PANCAN.gz")
    processing_brca_clinical_matrix_uri = ParameterString(name="BRCAClinicalMatrixURL", default_value="https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix")
    processing_train_test_split_ratio = ParameterString(name="TrainTestSplitRatio", default_value="0.2")
    processing_gene_count = ParameterString(name="GeneCount", default_value="20000")
    
    training_min_child_weight = ParameterInteger(name="MinChildWeight", default_value=5)
    training_num_round = ParameterInteger(name="NumRound", default_value=25)
    training_max_depth = ParameterInteger(name="MaxDepth", default_value=3)
    training_scale_pos_weight = ParameterFloat(name="ScalePosWeight", default_value=9.0)
    training_subsample = ParameterFloat(name="Subsample", default_value=0.9)
    test_accuracy_threshold = ParameterFloat(name="testAccuracy", default_value=0.8)

    registration_model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    ####################################################
    # Define Instance Types
    ####################################################    

    processing_instance_type = "ml.m5.xlarge"
    training_instance_type = "ml.m5.xlarge"

    ####################################################
    # Define Data Processing Step
    ####################################################    
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/her2-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="PreprocessHER2Data",
        processor=sklearn_processor,
        cache_config=CacheConfig(enable_caching=True, expire_after="30dT5h"),
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=[
            "--brca_clinical_matrix_url",
            processing_brca_clinical_matrix_uri,
            "--hiseq_url",
            processing_hiseq_uri,
            "--train_test_split_ratio",
            processing_train_test_split_ratio,
            "--gene_count",
            processing_gene_count,
            "--create_test_data"
        ],
    )

    ####################################################
    # Define Training Step
    ####################################################  
    
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/her2-train",
        sagemaker_session=sagemaker_session,
        role=role,
        rules=[Rule.sagemaker(rule_configs.create_xgboost_report())]
    )
    xgb_train.set_hyperparameters(
        objective="binary:logistic",
        booster="gbtree",
        eval_metric="error",
        min_child_weight=training_min_child_weight,
        num_round=training_num_round,
        max_depth=training_max_depth,
        scale_pos_weight=training_scale_pos_weight,
        subsample=training_subsample
    )
    step_train = TrainingStep(
        name="TrainHER2Model",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    ####################################################
    # Define Evaluation Step
    ####################################################      
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-her2-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="HER2EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateHER2Model",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    ####################################################
    # Define Conditional Registrstion Steps
    ####################################################  
    
    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    step_register = RegisterModel(
        name="RegisterHER2Model",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=registration_model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=test_accuracy_threshold,
    )
    step_cond = ConditionStep(
        name="CheckHER2Evaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    ####################################################
    # Create Pipeline
    ####################################################  
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_hiseq_uri,
            processing_brca_clinical_matrix_uri,
            processing_train_test_split_ratio,
            processing_gene_count,
            training_min_child_weight,
            training_num_round,
            training_max_depth,
            training_scale_pos_weight,
            training_subsample,
            test_accuracy_threshold,
            registration_model_approval_status,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
