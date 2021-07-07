# Image Style Transfer Using Convolutional Neural Networks
A Keras Implementation of [Image Style Transfer Using Convolutional Neural Networks, Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)  

The goal of this paper is to transfer styles from the source image while preserving the semantic content of the target image.

![example](images/example.png)

# Style Transfer
To transfer the style of <img src="https://latex.codecogs.com/svg.image?\vec{a&space;}"/>(style image) onto <img src="https://latex.codecogs.com/svg.image?\vec{p&space;}"/>(content image), we can define a loss function as follows:  

<img src="https://latex.codecogs.com/svg.image?L_{total}(\overrightarrow{p},&space;\overrightarrow{a},&space;\overrightarrow{x})=&space;\alpha&space;L_{content}(\overrightarrow{p},&space;\overrightarrow{x})&space;&plus;&space;\beta&space;L_{style}(\overrightarrow{a},&space;\overrightarrow{x})"/>

<img src="https://latex.codecogs.com/svg.image?\vec{x&space;}"/> is the generated image. α and β are the weighting factors for content and style reconstruction.  L<sub>content</sub> is how similar <img src="https://latex.codecogs.com/svg.image?\vec{p&space;}"/> and <img src="https://latex.codecogs.com/svg.image?\vec{x&space;}"/> are in their content representation. L<sub>style</sub> is how similar <img src="https://latex.codecogs.com/svg.image?\vec{a&space;}"/> and <img src="https://latex.codecogs.com/svg.image?\vec{x&space;}"/> are in their style representation.

In this paper, use CNN(VGG19) to generate images. Images can start from random image or content image. With each training, the resulting image is produced by updating the image with a smaller loss function. This is the method of updating the image directly, which has the disadvantage of taking a long time. This paper stated that it takes an hour to create a 512 x 512 image with an Nvidia K40 GPU.

# L<sub>content</sub>
The activation value in a specific layer of the VGG19 network was defined as a **content representation**. And **content loss** can be expressed as the difference between the two image(content, generated) representations.  

Content loss is defined as the squared error loss between two feature representations as follows:  

<img src="https://latex.codecogs.com/svg.image?L_{content}(\overrightarrow{p},&space;\overrightarrow{x},&space;l)&space;=&space;\frac{1}{2}\sum_{i,&space;j}^{}(F_{ij}^{l}-P_{ij}^{l})^2"/>

P<sup>l</sup> and F<sup>l</sup> their respective feature representation in layer l.  

The image is updated so that both images have the same content representation value. As a deeper layer is used, specific pixel information is lost, and if a lower layer is used, a result similar to content can be obtained. Below is an example image created using the activation values of conv4_2 and conv_2_2.  

![content_layer](images/content_layer.png)

# L<sub>style</sub>
In this paper, **Style representation** is defined as a correlation between different features. These feature correlations are given by the Gram matrix.

Style loss is defined as the squared error loss between two style representations as follows:  

<img src="https://latex.codecogs.com/svg.image?E_{l}=\frac{1}{4&space;N_{l}^{2}&space;M_{l}^{2}}&space;\sum_{i,j}^{}(G_{ij}^{l}&space;-&space;A_{ij}^{l})^{2}"/>

and the total style loss is  

<img src="https://latex.codecogs.com/svg.image?L_{style}(\overrightarrow{a},&space;\overrightarrow{x})=\sum_{l=0}^{L}w_{l}E_{l}" title="L_{style}(\overrightarrow{a}, \overrightarrow{x})=\sum_{l=0}^{L}w_{l}E_{l}" />

The image is updated so that both images have the same style representation. This is done by using gradient descent from a white noise image to minimise the mean-squared distance between the entries of the Gram matrices from the original image and the Gram matrices of the image to be generated.

# Result
- content image
<img src="codes/dataset/paris.jpg" width="400">

- style image
<img src="codes/dataset/starry_night.jpg" width="400">

- result (content_weight: 8e-4, style_weight=8e-1)
<img src="codes/result/result_3000_0.000800_0.800000.png" width="400">

# Difference Between Paper and Implementation
- Use ADAM optimizer instead of L-BFGS
- Use maximum pooling average pooling(I couldn't find a way to easily replace the corresponding layer in keras)
