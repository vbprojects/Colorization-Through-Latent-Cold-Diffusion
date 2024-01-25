The following is a capstone project done for the class DS 440 at Penn State University

The project structure is as follows,

VAE_Training.ipynb is the VQVAE training notebook

colddif.ipynb is the Latent Cold Diffusion model training algorithm

lcdm_sampling.ipynb is the results notebook

# Introduction

*Define the problem you are tackling, introduce the challenges, and provide an overview of your solution. State clearly what you are demonstrating, how, and provide a teaser of your results.*


We seek to be able to recolor images through the use of Cold Diffusion in the Latent Space of a Variational Auto-Encoder (VAE). Challenges in this field are generally within implementation. Latent Space Recolorization models are novel as most methods have been Convolutional Neural Networks (CNN) working within the pixel space of an image. Using the latent space of a VAE lowers the computational cost of our training algorithm and extends possible training of the Cold Diffusion model on the embeddings of a better pre-trained VAE. While we tackle Image Colorization, our results may indicate that many image degradations (blur, noise, snowification, etc.) are possible to invert solely through transformations in latent spaces.

![image](https://github.com/vbprojects/Colorization-Through-Latent-Cold-Diffusion/assets/66980754/edaf9c9f-93bb-4f72-8bf7-b7479d74bff6)

# Related Work

*How does your work relate to what exists in the real world? What state-of-the-art solutions exist and what are their pros and cons? Briefly describe your choice of solutions and why you chose them.*

Many of the assumptions of diffusion probabilistic models stem from Langevin dynamics, which imposes restrictions and computational burdens on the class of model. A recent paper, Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise, challenged this assumption by removing the stochasticity of the process, a fundamental assumption in the original paper. However, this paper implemented diffusion within pixel space, but most state-of-the-art methods operate in the latent space of a VQ-VAE. Additionally, the paper Improved Diffusion-based Image Colorization via Piggybacked Models implemented a colorization scheme based on text-to-image in the latent space of a CVAE. The model does not implement Cold Diffusion which could be a more direct approach to colorizing a specific image. 

# Methods

*State clearly what you implemented and how.*

We implemented a Vector Quantized Variational Auto-Encoder (VQ-VAE, https://arxiv.org/abs/1711.00937) on the Large-scale CelebFaces Attributes (CelebA, https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset rescaled to be between -1 to 1 and a 64 by 64 images with 3 channels. We then used a cold latent diffusion model (https://arxiv.org/abs/2208.09392) with a residual architecture conditioned by timestep with sinusoidal time embeddings. 

We trained the VQ-VAE with 32 embeddings and a codebook with 512 embeddings. The dimensions of our latent space are 32 by 16 by 16. In order to generate our grayscale data, we averaged the color channels of the input image. The loss function during training was a combination of reconstruction loss for which we chose Mean Squared Error (MSE), and Vector-Quantized Loss (VQ loss) for generating an efficient codebook.

Our Latent Cold Diffusion Model (LCDM) is 8 layers deep with Rectified Linear Units (ReLU, https://arxiv.org/abs/1803.08375) as the activation function. Conditioning was applied to the model by using the hammard product of the sinusoidal time embeddings and each of the layer's outputs, and inputs to the embedding were normalized between 0 and 1 based on the maximum time steps which were 50. We trained the model to reconstruct paired color latent embeddings from grayscale latent embeddings. We first flattened the pairs of image embeddings of the autoencoder to a 8192 dimension vector, then trained the model to construct color embeddings from grayscale embeddings. The loss function was MSE between predicted and true color embeddings. We assigned each embedding in the batch a step between 0 and the max step. We then interpolated grayscale embedding linearly towards the colorized embedding by the normalized step. For example, 50% if the step was 25, 0% if 50, and 100% if 0. We both used that position as a starting point for the LCDM and conditioned the model by the step. 

We chose two sampling methods for the LCDM, oneshot and the improved sampling method outlined in the paper Cold Diffusion: Inverting Abrtritary Transformations (https://arxiv.org/abs/2208.09392). Oneshot involved the model reconstructing the colorized embedding from the grayscale embedding in one step, acting more like a residual network than an LCDM. The sampling second sampling method from the paper is as follows

![image](https://github.com/vbprojects/Colorization-Through-Latent-Cold-Diffusion/assets/66980754/ba9c2e91-0b59-46ee-87d2-0cacd7daec45)

R in this image is the LCDM, D is the linear projection, in this case from x_s to x_0 by s normalized to be between 0 and 1 by max steps. For both models, we used an AdamW optimizer (https://arxiv.org/abs/1711.05101) and trained for 100 epochs.

# Evaluation

*State clearly how you evaluated things. Identify the datasets used and why they were chosen. Provide the results and discuss the results obtained.*

We chose the CelebA dataset because we thought training a model on the dataset would fit three criteria. 

●	The dataset should make model training within computational reach

●	The dataset should have a diverse set of features and be reasonably complex

●	The dataset should highlight a practical element of recolorization

We had originally thought of using the Flowers102, Cifar10, or MNIST dataset, however, we decided on CelebA as it fit not only all of our criteria because a large part of the social element of recolorizing photos centers around people.

While we would have preferred to use Flechet Inception Distance (FID), we chose MSE and Mean Absolute Percentage Error (MAPE) due to time constraints. Specifically reconstruction loss (MSE) for evaluating the VQ-VAE and MAPE for between predicted colorized embeddings and true colorized embeddings. 

MSE of 0.028 for the VQ-VAE. MAPE of 14% for Improved Reconstruction algorithm and 16% for OneShot.

We trained the best VQ-VAE model we could based on time constraints, better VQ-VAE models would lead to better reconstructions, however, that is not the aim of our project. We set out to show that Cold Diffusion Models operating in the latent space of a VQ-VAE could be effective for Image Colorization, and our results clearly show that this model and training algorithm can recolor images. However, the difference between OneShot and the Improved Sampling Method could indicate that a simpler feed-forward or unconditioned residual network may be able to match the performance of our LCDM. 

# Discussion

*Discuss what worked and what did not in your project and what needs to be done in the future.*

Our results are in line with the theoretical understanding of both VQ-VAEs and Cold Diffusion Models. VQ-VAE assembles a codebook of discrete features relating to an image and outputs that discrete codebook in the latent embedding. In the process of training, the model most likely picked up on the very clear semantic differences of colorized and grayscale images and some part of the codebook represents that. Thus, a small amount of the latent code would need to change in order to change the reconstruction of the image from grayscale to color. These small changes may make only a fraction of the full latent space. Further work should be done to identify parts of the embedding corresponding to colorization.

Additionally, linear projection from one embedding to another is a linear degradation, and a Cold Diffusion Model should be able to invert the transformation. Our choice of model is not standard but was easy to implement and fast to train. A U-Net architecture conditioned on timestep with sinusoidal embeddings would be closer to the State of the Art.

Moving forward, we need better evaluation metrics for image reconstruction. For example, Flechet Inception Distance. Furthermore, the VQ-VAE should be updated to a VQ-VAE 2 and a U-Net Architecture should be used instead of a Residual Dense Network for the Latent Cold Diffusion Model.

This method could be expanded to working on embeddings of pre-trained VQ-VAE 2 models, particularly on ImageNet. Even if the models were not explicitly trained to reconstruct a proportional amount of grayscale to color images, Image reconstruction quality could be significantly increased compared to our work.

# Conclusion

*What conclusions can we draw from your project?*

It is clear to see the effectiveness of Latent Cold Diffusion Models on Image Colorization. Advancements in this technique should also apply to other image degradations like noise, blur, masking, pixelation, and snowification. Additionally, the technique may be effective for any image-to-image translation problems. Latent Diffusion Models already represent a democratization of image generation. State of the Art Generative Adversarial Models, while fast, are difficult to train for the average consumer and researcher. Reversing linear degradations with semantic meaning through Latent Cold Diffusion models can further democratize image synthesis and contribute to locally trainable open-source projects.


