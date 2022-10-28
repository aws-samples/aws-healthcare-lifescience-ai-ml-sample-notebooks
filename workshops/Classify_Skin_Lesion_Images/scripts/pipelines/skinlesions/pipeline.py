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
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from time import strftime

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
    model_package_group_name="LesionClassifierPackageGroup",
    pipeline_name="LesionClassifierPipeline",
    base_job_prefix="LesionClassifier",
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
    # Define Data Processing Step
    ####################################################    
    pre_processor = sagemaker.processing.FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version='1.0-1',
        command=['python'],
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        sagemaker_session=PipelineSession()
    )

    # Create a lazy initialization of the processor run that will wait to run during the pipeline execution
    processor_args = pre_processor.run(
        job_name=f"skin-lesion-image-processing-job-{strftime('%Y-%m-%d-%H-%M-%S')}",
        code="preprocess.py",
        source_dir=BASE_DIR,
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{default_bucket}/{base_job_prefix}/data/train/",
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/val",
                destination=f"s3://{default_bucket}/{base_job_prefix}/data/val/",
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{default_bucket}/{base_job_prefix}/data/test/",
            ),
        ]
    )
    
    # Use the lazy processing run to define a ProcessingStep
    processing_step = ProcessingStep(
        name="LesionImageProcessingStep",
        step_args=processor_args,
    )    

    ####################################################
    # Define Training Step
    ####################################################  
    
    model_id, model_version = "tensorflow-ic-imagenet-mobilenet-v2-100-224-classification-4", "*"
    training_instance_type = "ml.p3.2xlarge"
    mobilenet_job_name = f"mobilenet-Training-Job"
    from sagemaker import image_uris, model_uris, script_uris, hyperparameters

    # Retrieve the Docker image uri
#     train_image_uri = image_uris.retrieve(
#         model_id=model_id,
#         model_version=model_version,
#         image_scope="training",
#         instance_type=training_instance_type,
#         region=None,
#         framework=None)
    train_image_uri = "763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.8-gpu-py3"

    # Retrieve the training script uri
    train_source_uri = script_uris.retrieve(
        model_id=model_id, 
        model_version=model_version, 
        script_scope="training")

    # Retrieve the pretrained model artifact uri for transfer learning
    train_model_uri = model_uris.retrieve(
        model_id=model_id, 
        model_version=model_version, 
        model_scope="training")

    # Retrieve the default hyper-parameter values for fine-tuning the model
    hyperparameters = hyperparameters.retrieve_default(
        model_id=model_id, 
        model_version=model_version
    )

    # Override default hyperparameters with custom values
    hyperparameters["epochs"] = "3"
    hyperparameters["batch_size"] = "70"
    hyperparameters["learning_rate"] = 0.00010804583232953079
    hyperparameters["optimizer"] = 'rmsprop'

    # Specify S3 urls for input data and output artifact
    training_dataset_s3_path = f"s3://{default_bucket}/{base_job_prefix}/data/train"
    validation_dataset_s3_path = f"s3://{default_bucket}/{base_job_prefix}/data/val"
    s3_output_location = f"s3://{default_bucket}/{base_job_prefix}/output"

    # Specify what metrics to look for in the logs
    training_metric_definitions = [
        {"Name": "val_accuracy", "Regex": "- val_accuracy: ([0-9\\.]+)"},
        {"Name": "val_loss", "Regex": "- val_loss: ([0-9\\.]+)"},
        {"Name": "train_accuracy", "Regex": "- accuracy: ([0-9\\.]+)"},
        {"Name": "train_loss", "Regex": "- loss: ([0-9\\.]+)"},
    ]

    # Create estimator
    tf_ic_estimator = sagemaker.estimator.Estimator(
        base_job_name = mobilenet_job_name,
        role=role,
        image_uri=train_image_uri,
        source_dir=train_source_uri,
        model_uri=train_model_uri,
        entry_point="transfer_learning.py",
        instance_count=1,
        instance_type=training_instance_type,
        hyperparameters=hyperparameters,
        output_path=s3_output_location,
        enable_sagemaker_metrics=True,
        metric_definitions=training_metric_definitions,
        sagemaker_session=PipelineSession()
    )

    # Create a lazy initialization of the training run that will wait to run during the pipeline execution
    training_args = tf_ic_estimator.fit(
        inputs = {
            "training": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,content_type="text/csv"
            )
        }
    )

    # Use the lazy training run to define a TrainingStep
    training_step = TrainingStep(
        name="LesionClassifierTrainingStep",
        step_args=training_args
    )
    ####################################################
    # Define Evaluation Step
    ####################################################      

    
    eval_args = pre_processor.run(
        job_name=f"skin-lesion-evaluation-job-{strftime('%Y-%m-%d-%H-%M-%S')}",
        code="evaluate.py",
        source_dir=BASE_DIR,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ]
    )    
    evaluation_report = PropertyFile(
        name="LesionClassifierEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluation_step = ProcessingStep(
        name="EvaluateLesionClassifierModel",
        step_args=eval_args,
        property_files=[evaluation_report],
    )

    ####################################################
    # Define Conditional Registration Steps
    ####################################################  
    
    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    step_register = RegisterModel(
        name="RegisterLesionClassifierModel",
        estimator=tf_ic_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=0.5,
    )
    conditional_step = ConditionStep(
        name="CheckLCEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    ####################################################
    # Create Pipeline
    ####################################################  
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[processing_step, training_step, evaluation_step, conditional_step],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
