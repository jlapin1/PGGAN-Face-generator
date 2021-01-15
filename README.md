# PGGAN-Face-generator

This is an independent project in which I try to reproduce the model from [1] and partially reproduce the reported results as well.
In short, this method of facial generation is to:

1. Train a GAN, generator and discriminator, first at low resolution until convergence
2. Increase the resolution and gradually burn in the new layers
3. Train at new resolution until convergence
4. Repeat until final resolution is reached

In the words of the authors:

"This incremental nature allows the training to first discover large-scale structure of the
image distribution and then shift attention to increasingly finer scale detail, instead of having to learn
all scales simultaneously."

In the original paper the authors report results up to 1024x1024 resolution using the CelebA-HQ dataset.
The size of the dataset (~90 GB) prevented me from reproducing photos at that resolution, so I modified the model to produce 128
resolution instead. I used the CelebA dataset consisting of ~210k images of celebrities.

Instead of the WGAN-GP loss function from the paper, I use Least squares loss (which the authors use partially) with the zero
centered R1 gradient penalty on real data. The gradient penalty was 0.1 up through 32x32, but then needed to be increased slowly
thereafter as the image quality degraded otherwise. Furthermore resolution's above 32 needed longer training in order to mitigate
artifacts and incongruencies in the generated images.

Results shown are the different resolutions side by side in a png, as well as a gif showing the animated training progression.

[1] Karras, T., Aila, T., Laine, S., Lehtinen, J. (2018) Progressive Growing of GANs for Improved Quality, Stability, and Variation
