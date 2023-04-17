params.s3_input=""
params.rf_design_output=""
params.esmfold_output=""
params.outdir= ""


project_dir = projectDir

process run_rf_design {

    output:
    path 'batch_job_ids.txt'

    """
    python ${project_dir}/bin/run_rfdesign.py --input_s3_uri ${params.s3_input} --output_s3_uri ${params.rf_design_output} --num_sequences_to_generate 3 > batch_job_ids.txt
    """
}

process wait_for_batch{
   
    input:
    path 'batch_job_ids.txt'

    output: 
    path "hello_from_waiter.txt"


   """
   python ${project_dir}/bin/wait_for_batch.py batch_job_ids.txt > hello_from_waiter.txt
   """

}

process run_esmfold {
    
    input:
    path x

    output:
    file 'hello_from_esmfold.txt'

    """
python /root/bin/get_fastas.py ${params.rf_design_output} |while read line; do python /root/bin/run_esmfold.py \$line ${params.esmfold_output}; done > hello_from_esmfold.txt
    """

}

workflow {
  run_rf_design|wait_for_batch|run_esmfold
}
