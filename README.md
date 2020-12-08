# Deep Learning Project: Unsupervised Anomaly Detection on X-Ray Images
## Project Summary
In our project we plan on using Generative Adversrial Network (GAN). A GAN is a class of AI systems, where Two neural organizations challenge with one another in a game (as a lose-lose situation, where one specialist's benefit is another specialist's misfortune). For our project we will be using two GANs, AlphaGAN and AnoGAN in-order to detect anomalies in the X-ray images. 
1. AlphaGAN: The generator of this network is a convolutional encoder-decoder network that is trained both with help of the ground-truth alphas as well as the adversarial loss from the discriminator, and the discriminator is a patchGAN Discriminator.
2. AnoGAN: the firstly proposed method using GAN for anomaly detection. The generator of GAN is trained to produce patches and fit the data distribution. Based on the second loss, the generator takes not only the information to fool the discriminator but the rich information of the feature representation         

## Summary
In this repository you will find the files required to train multiple different models in-order to find anomalies in x-ray images.

In this repository you will find
  - Jupyter Notebooks / Google Colab Notebooks: Contains colab for AlphaGAN Training; f-AnoGAN Training; Inference Code for Anomaly Detection. 
  - Delivarables
         1. Proposal [pdf](https://github.com/plodha/CMPE-297-DeepLearning/blob/main/Deliverables/Project%20Proposal%20-%20TheMeanSquares.pdf)
         2. Project Report [pdf](https://github.com/plodha/CMPE-297-DeepLearning/blob/main/Deliverables/X-Ray%20Anomaly%20Detection%20Project%20Paper.pdf)
         3. Presentation [pdf](https://github.com/plodha/CMPE-297-DeepLearning/blob/main/Deliverables/CMPE%20297%20Deep%20Learning%20Project.pdf)
  - WebApp (frontend, backend)
  - TFX
