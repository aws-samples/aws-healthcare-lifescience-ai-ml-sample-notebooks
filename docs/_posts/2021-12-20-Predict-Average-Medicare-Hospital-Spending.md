---
layout: post
title:  Predict Average Medicaree Hospital Spending
date:   2021-12-20 00:00:00 +0000
categories: Workshops
---

Medicare is a national health insurance program, administered by the Center for Medicare and Medicaid Services (CMS). This is a primary health insurance for Americans who are aged 65 and older. Medicare has published historical data showing hospital’s average spending for Medicare Part A and Part B claims based on different claim types and claim periods covering 1 to 3 days prior to hospital admission up to 30 days after discharge from hospital admission. These hospital spending are price standardized and non-risk adjusted, since risk adjustment is done at the episode level of the claims spanning the entire period during the episode. The hospital average costs are listed against the corresponding state level average cost and national level average cost.

In this notebook, the data is used to build a machine learning model using Amazon SageMaker built-in Linear Learner algorithm, which predicts average hospital spending cost based on the average state level spending and average national level spending. The predicted cost can be used for purposes of budget and for negotiating pricing with the hospitals. From the hospital’s perspective, the predicted average hospital spending provides visibility to claim financials that can be used by the hospitals to increase their efficiency and level of care.

Resources :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks/blob/main/workshops/Medicare_Hospital_Cost_Prediction/Jupyter_Notebook_Medicare_Hospital_Cost_Prediction.ipynb){:target="_blank"}{:rel="noopener noreferrer"}
