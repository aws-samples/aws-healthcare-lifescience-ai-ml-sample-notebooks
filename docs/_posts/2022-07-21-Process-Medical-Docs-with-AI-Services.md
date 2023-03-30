---
layout: post
title:  Medical Document Processing with Amazon Textract, Amazon Comprehend, and Amazon Comprehend Medical
date:   2022-07-21 00:00:00 +0000
categories: Workshops
author: 
    name: Brian Loyal
---

In this notebook, we will walkthrough on how to build a data processing pipeline that will process electronic medical reports (EMR) in PDF format to extract relevant medical information by using the following AWS services:

* Textract: To extract text from the PDF medical report
* Comprehend: To process general language data from the output of Textract.
* Comprehend Medical: To process medical-domain information from the output of Textract.

Resources:

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Process_HCLS_Docs_Using_AI_Services/Process-Medical-Documents.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
