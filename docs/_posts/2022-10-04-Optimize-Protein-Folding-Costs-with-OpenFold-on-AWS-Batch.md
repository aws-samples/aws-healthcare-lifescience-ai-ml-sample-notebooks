---
layout: post
title:  "Optimize Protein Folding Costs with OpenFold on AWS Batch"
date:   2022-10-04 00:00:00 +0000
categories: Blogs 
author: 
    name: Brian Loyal
---

Knowing the physical structure of proteins is an important part of the drug discovery process. Machine learning (ML) algorithms like AlphaFold v2.0 significantly reduce the cost and time needed to generate usable protein structures. These projects have also inspired development of AI-driven workflows for de novo protein design and protein-ligand interaction analysis.

Researchers have used AlphaFold to publish over 200 million protein structures. However, newer algorithms may provide cost or accuracy improvements. One example is OpenFold, a fully open-source alternative to AlphaFold, optimized to run on widely available GPUs.

In this post, we build on prior work to describe how to orchestrate protein folding jobs on AWS Batch. We also compare the performance of OpenFold and AlphaFold on a set of public targets. Finally, we will discuss how to optimize your protein folding costs.

*Resources* :

* [![Github](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png){: width="50" }](https://github.com/aws-solutions-library-samples/aws-batch-arch-for-protein-folding){:target="_blank"}{:rel="noopener noreferrer"}
* [AWS HPC Blog](https://aws.amazon.com/blogs/hpc/optimize-protein-folding-costs-with-openfold-on-aws-batch/){:target="_blank"}{:rel="noopener noreferrer"}
