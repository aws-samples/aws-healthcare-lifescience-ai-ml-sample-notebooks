# ESMFold on SageMaker

Jupyter notebook describing how to run the ESMFold protein folding algorithm in SageMaker.

Understanding the structure of proteins like antibodies is important for understanding their function. However, it can be difficult and expensive to do this in a laboratory. Recently AI-driven protein folding algorithms have enabled biologists to predict these structures from their aminio acid sequences instead.

In this notebook, we will use the [ESMFold](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1) protein folding algorithm to predict the structure of Herceptin (Trastuzumab), an important breast cancer therapy. Herceptin is a [monoclonal antibody](https://www.cancer.org/treatment/treatments-and-side-effects/treatment-types/immunotherapy/monoclonal-antibodies.html) (mAb) that binds to the HER2 receptor, inhibiting cancer cell growth. The following diagram shows several of the common elements of monoclonal antibodies.

![A diagram of the major structural elements of an antibody](img/antibody.png)

In this notebook, we'll focus on predicting the structure of the heavy chain region.
