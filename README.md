# Neural Style Transfer
This project is developed by Ömer Ünlüsoy, Jonathan Menssen, and Jorge Cerrada for the ICFP M2 Machine Learning course. 
- Neural Style Transfer is a Computer Vision subfield that deal with the extraction of the artistic style of one image and transferring it into another while preserving the semantic content using Convolutional Neural Networks (CNNs). 
- This project implements the __A Neural Algorithm of Artistic Style__ algorithm proposed by Gatys et al. in their seminal paper [[1]](#1). 


### Folder Structure
- __`Gatys_Image_Style_Transfer_CVPR_2016_paper`__ : Seminal paper by Gatys et al. with their 'A Neural Algorithm of Artistic Style' algorithm. 
- __`Style_Transfer.ipynb`__ : Jupyter Notebook containing the full Neural Style Transfer implementation. It is meant to be run on Google Colab (hopefully with A100) since it includes more than 15 runs for experiments and examples, each with ~300 epochs with LBFGS Optimizer. It already includes all the outputs (took half an hour with A100) from a single run.
- __`core.py`__ : The core implementation of Neural Style Transfer without any overheads. It includes the full complexity of the project and should be used to understand the code as well as single example runs. Do not forget to change the 2 image paths in the code for an example run. 
- __`presentation.pdf`__ : The presentation of the Neural Style Transfer project. 
- __`Collections folder`__ : It includes necessary images for Style_Transfer.ipynb to run. 
- __`run_01-07-33_12-03-2025_UTC+0000 folder`__ : The latest run of Style_Transfer.ipynb including target images as well as training videos of several content and style images with different hyperparameters.


### A Neural Algorithm of Artistic Style
A Neural Algorithm of Artistic Style is a Style Transfer algorithm proposed by Gatys et al. in their seminal work [[1]](#1). 
- It uses a Convolutional Neural Network (CNN), a __VGG-19 architecture__ pre-trained on ImageNet dataset, to extract image features.
- __LBFGS Optimizer__ runs on the target image rather than on the model (all model parameters are frozen).
- Loss function is calculated based on the Content Features of content image and Style Features of style image. 
- Target image tries to optimize its contents close to the content image and its style close to the style image.

### Loss
Loss function is the weighted sum of Content Loss $\mathcal{L}_c$ and Style Loss $\mathcal{L}_s}$.
- $\vec{x}$ : target image
- $\vec{p}$ : content image
- $\vec{a}$ : style image
&nbsp;
- $F^l$ : Content Features of target image    &nbsp; ($l$ runs through Content Layers)
- $P^l$ : Content Features of content image 	
- $v^l$ : Content weights
&nbsp;
- $F^L$ : Style Features of target image  &nbsp; ($L$ runs through Style Layers)   
- $A^L$ : Style Features of style image
- $w^L$ : Style weights
- $G(X)$ : Gram matrix of $X$


$$ \mathcal{L}_c}(\vec{x}, \vec{p}, \vec{v}) = \sum_l v^l  ( F^l - P^l )^2 $$

$$ \mathcal{L}_s}(\vec{x}, \vec{a}, \vec{w}) = \sum_L w^L  ( G(F^L) - G(A^L) )^2 $$

$$\mathcal{L}_{\textit{total}} (\alpha, \beta) = \alpha \mathcal{L}_c} + \beta \mathcal{L}_s}$$



### Hyperparameters
- __`Weight ratio (alpha / beta)`__ : we run an experiment with different style_weights = [1e-2, 1e-1, 1, 1e1, 1e2, 1e4].
- __`Learning rate`__	: our experiment suggests an optimal value of 0.1 - 0.3 for LBFGS.
- __`Epoch`__	: our experiment suggests an optimal value of 200 - 300 for LBFGS based on the the images.

- __`Selected Content layers`__ : Gatys et al. paper proposes 'conv4_2'.
- __`Content weights`__ : Gatys et al. paper proposes 1.
- __`Selected Style layers`__ : Gatys et al. paper proposes 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'.
- __`Style weights`__ : Gatys et al. paper proposes [1e3/n**2 for n in [64, 128, 256, 512, 512]].


### References
<a id="1">[1]</a> L. A. Gatys, A. S. Ecker and M. Bethge, "Image Style Transfer Using Convolutional Neural Networks," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 2414-2423, [doi:10.1109/CVPR.2016.265](https://doi.org/10.1109/CVPR.2016.265).
