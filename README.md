# PGGAN-Face-generator

This is an independent project in which I try to reproduce the model from [1] and partially reproduce the reported results as well.
In the original paper the authors report results for a celebrity face generator up to resolution 1024 using the CelebA-HQ dataset.
The size of the dataset (~90 GB) prevented me from reproducing photos at that resolution, so I modified the model to produce 128
resolution instead.

Instead of the WGAN-GP loss function from the paper, I use Least squares loss (which the authors use partially) with the zero
centered R1 gradient penalty on real data. The gradient penalty was 0.1 up through 32x32, but then needed to be increased slowly
thereafter as the image quality degraded otherwise. Furthermore resolution's above 32 needed longer training in order to mitigate
artifacts and incongruencies in the generated images.

Results shown are the different resolutions side by side in a png, as well as a gif showing the animated training progression.
