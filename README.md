# Stable Diffusion

## Credits & Acknowledgements
This project is a reimplementaion of [StableDiffusion-PyTorch] by [explainingai-code] (https://github.com/explainingai-code/StableDiffusion-PyTorch)
The code has been rewritten from scratch while maintaining the core concepts and functionalities of the original implementation.

## Features
- A Vector-Quantized Variational Autoencoder (VQ-VAE) to compress images into latent space and decoded latent data back into image space.
- A Latent Diffusion Model (LDM) that can accept class, image and/or text conditioning input to control the generated output.
- The LDM is trained under classifier-free guidance.
- Both are coded in such a way that can be modified using a config file.

## Description of files in dataset
- **MNIST_dataset.py** - Create a custom dataset for MNIST that return images / latent data and class conditions.
- **CELEB_dataset.py** - Create a custom datset for CELEB that return images / latent data and text & image conditions.

## Description of files in models
- **blocks.py** - Contain the codes for the construction of essential blocks for both autoencoder and diffusion models.
- **VQVAE.py** - Vector-Quantized Variational Autoencoder for compressing images into latent data or reconstruction from latent data.
- **discriminator.py** - A discriminator model that is used to evaluate the reconstruction performance of an autoencoder.
- **lpips.py** - A pretrained model that is used to evaluate the perception loss from the reconstruction result by an autoencoder (The pretrained weight can be located in the weights folder).
- **UNet.py** - The backbone of the Stable Diffusion Model.

## Description of files in scheduler
- **linear_noise_scheduler.py** - A linear noise scheduler that is needed in forward diffusion to add noise and in reverse diffusin to remove noise.

## Description of files in tools
- **train_VQVAE.py** - Trains a VQVAE model.
- **train_UNET.py** - Trains a Stable Diffusion (with and without conditions) with U-Net as the backbone.
- **infer_VQVAE.py** - To extract images in the form of latent data and reconstruct images from extracted latent data using VQVAE.
- **infer_UNET.py** - To generate images using a trained Stable Diffusion (with and without conditions).
- **infer_UNET.py** - To generate images using a Stable Diffusion (trained with both text and image conditions), but only with text OR image conditions.

## Description of files in utils
- **extract_mnist.py** - Extract MNIST data from csv files.
- **dataset_test.py** - To verify if the dataset is created correctly.
- **create_celeb_mask.py** - To compile all the mask images for the same image into one image. 
- **text_utils.py** - To get the required pretrained tokenizer and text model for the encoding of the text conditions.
- **diffusion_utils.py** - To load latent data and randomly drop conditions for training under classifier-free guidance.

## Results

- A) **Reconstruction of images using the latent data extracted by a VQVAE trained with MNIST**

![VQVAE_MNIST](./result/[1] VQVAE_MNIST reconstruction.png)

- B) **Reconstruction of images using the latent data extracted by a VQVAE trained with CELEB**

![VQVAE_CELEB](./result/[2] VQVAE_CELEB reconstruction.png)

- C) **Generation results using the Stable Diffusion, trained with MNIST under class conditions**
	- The input class conditions for this sample is [9, 4, 1, 8, 6, 1, 8, 4, 1, 9, 8, 5, 2, 7, 0, 1, 8, 1, 7, 7, 1, 9, 6, 3, 0].

![LDM_MNIST_CLASS](./result/[6] MNIST class condition.png)

- D) **Generation results using the Stable Diffusion, trained with CELEB under text conditions**
	- The input text condition for this sample is [This person has black hair, and oval face. She is attractive and wears heavy makeup.]

![LDM_CELEB_TEXT](./result/[5] CELEB text condition.png)

- E) **Generation results using the Stable Diffusion, trained with CELEB under image conditions**

![LDM_CELEB_IMAGE_MASK](./result/[4] CELEB image mask.png)
![LDM_CELEB_IMAGE](./result/[4] CELEB image.png)

- F) **Generation results using the Stable Diffusion, trained with CELEB under text and image conditions**
	- The input text condition for this sample are 
		[She is smiling, and young and has high cheekbones, big nose, and mouth slightly open.] 
		[This man has bushy eyebrows and wears lipstick. He is young.]

![LDM_CELEB_IMAGE_TEXT_MASK](./result/[3] CELEB image text mask .png)
![LDM_CELEB_IMAGE_TEXT](./result/[3] CELEB image text.png)

- G) **Generation results using the Stable Diffusion, trained with CELEB under text and image conditions, but inferred by dropping image**
	- The input text condition for this sample is [She has high cheekbones, pointy nose, mouth slightly open, and rosy cheeks and wears lipstick, and necklace. She is young.]

![LDM_CELEB_DROP_IMAGE](./result/[8] CELEB drop image.png)

- H) **Generation results using the Stable Diffusion, trained with CELEB under text and image conditions, but inferred by dropping text**

![LDM_CELEB_DROP_TEXT_MASK](./result/[7] CELEB drop text mask.png)
![LDM_CELEB_DROP_TEXT](./result/[7] CELEB drop text.png)




































