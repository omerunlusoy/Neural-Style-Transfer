# Core functionality of Style_Transfer without distractions.
# written by Ömer Ünlüsoy, Jonathan Menssen, Jorge Cerrada
# CHECK content_image_path and style_image_path before running.

import torch
from torchvision import transforms, models

import numpy as np
import matplotlib.pyplot as plt

import os
import time
from datetime import timedelta
from PIL import Image
import argparse

# Determine the running path for default paths
running_path = os.path.realpath(__file__)
running_path = os.path.dirname(running_path)


# Set default paths and hyperparameters
default_content_image_path = os.path.join(running_path, "Collections", "HayleyWilliams.jpg")
default_style_image_path   = os.path.join(running_path, "Collections", "Bank.jpg")

# style content ratio: alpha represents content image weight and beta represents style image weight
default_style_weight = 1e1
content_weight = 1

default_learning_rate_Adam = 0.06
default_learning_rate_LBFGS = 0.2

default_adam_epoch = 0
default_LBFGS_epoch = 200

print_per = 10

# dictionary that holds the specific layer numbers where features will be extracted. You can play with them.
# Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv4_2, Conv5_1
layers = {'0': 'conv1_1',    # style extraction
		  '5': 'conv2_1',    # style extraction
		 '10': 'conv3_1',    # style extraction
		 '19': 'conv4_1',    # style extraction
		 '21': 'conv4_2',    # content extraction
		 '28': 'conv5_1'}    # style extraction

content_layer = 'conv4_2'

# assign weight to each style layer for representation power (early layers have more style)
# one of the hyperparameters to be experimented with
style_weights = {'conv1_1': 1e3/64**2,
				 'conv2_1': 1e3/128**2,
				 'conv3_1': 1e3/256**2,
				 'conv4_1': 1e3/512**2,
				 'conv5_1': 1e3/512**2}

# specifies run device for more optimum runtime
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  

def image_convert_to_numpy(tensor):
	""" Converts image from the torch tensor to numpy array.
	1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels

	Args:
		tensor (torch image): image to be converted

	Returns:
		numpy image: converted numpy image
	"""

	# clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
	image = tensor.clone().detach().cpu().numpy()
	image = image.squeeze()
	image = image.transpose(1, 2, 0)
	# print(image.shape)                                                                            # (28, 28, 1)
	# denormalize image
	image = image * np.array((0.5,)) + np.array((0.5,))
	image = image.clip(0, 1)
	return image


def load_image(path, max_size=600, shape=None):
	""" Loads the image from the given path and resizes for model compatibility.

	Args:
		path (String): Image path
		max_size (int, optional): Maximum image size. Defaults to 600.
		shape (image shape, optional): image shape. Defaults to None.

	Returns:
		torch image: loaded image
	"""
	image = Image.open(path).convert('RGB')
	size = max(image.size)

	if size > max_size:
		size = max_size

	if shape is not None:
		size = shape

	# transform image to be compatible with the model
	transform = transforms.Compose([
				transforms.Resize(size),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,))])

	image = transform(image).unsqueeze(0)  # to add extra dimensionality
	return image


def save_image(image):
	""" Saves the given image with the specified name (full path)

	Args:
		image (torch image): image to be saved
		name (String): full path of the image
	"""

	file_name = str(running_path) + "/target_image_" +  str(os.path.splitext(os.path.basename(content_image_path))[0]) + "_" + str(os.path.splitext(os.path.basename(style_image_path))[0]) + "_adam_epoch=" + str(adam_epoch) + "_LBFGS_epoch=" + str(LBFGS_epoch) + "_lr=" + str(learning_rate_Adam) + "_" + str(learning_rate_LBFGS) + "_style_weight=" + str(style_weight) + ".jpg"
	print("Target image is saved at", file_name)
	plt.imsave(file_name, image_convert_to_numpy(image), dpi=500)


# returns pre-trained VGG19 model
def get_model():
	# VGG 19 pre-trained model
	model = models.vgg19(weights="DEFAULT").features

	# freeze parameters
	for param in model.parameters():
		param.requires_grad = False

	# send model to device
	model.to(device=device)
	return model

	
# extracts the features from image using model
def get_features(model, image):
 	# dict that will store the extracted features
	features = {}
	# iterate through all layers and store the ones in the layers dict
	for name, layer in model._modules.items():	
  		# run image through all layers
		image = layer(image)
  
		# fetch and store the features if the layer is in layers (our interest)
		if name in layers:
			features[layers[name]] = image
	return features


# Gram Matrix = V(T)*V  T: Transpose
def gram_matrix(tensor):
	# takes 4D image tensor
	# reshape the tensor
	_, d, h, w = tensor.size()
	tensor = tensor.view(d, h * w)
	gram = torch.mm(tensor, tensor.t())
	return gram


def train(model, content_image, style_image, adam_epoch=1000, LBFGS_epoch=200):
	
 	# Pre-calculate content and style features
	content_features = get_features(model, content_image)
	style_features = get_features(model, style_image)

	# Compute style gram matrices
	style_grams = { layer: gram_matrix(style_features[layer]) for layer in style_features }

	# Initialize the target image (the one we optimize)
	target_image = content_image.clone().requires_grad_(True).to(device=device)
	height, width, channels = image_convert_to_numpy(target_image).shape

	global_step = 0  # counts total training iterations
	start_training_time = time.time()
 
	# -------------------------
	# Phase 1: Adam Optimization
	# -------------------------
	optimizer = torch.optim.Adam([target_image], lr=learning_rate_Adam)
	
	for i in range(1, adam_epoch + 1):
		# Forward pass: get target features
		target_features = get_features(model, target_image)

		# Content loss: mean squared error between content features
		content_loss = torch.mean( (target_features[content_layer] - content_features[content_layer]) ** 2 )

		# Style loss: comparing Gram matrices for each style layer
		style_loss = 0
		for style_layer in style_weights:
			target_feature = target_features[style_layer]
			target_gram = gram_matrix(target_feature)
			style_gram = style_grams[style_layer]
			current_style_loss = style_weights[style_layer] * torch.mean( (target_gram - style_gram) ** 2 )
			
   			# Normalize style loss per layer
			_, d, h, w = target_feature.shape
			style_loss += current_style_loss / (d * h * w)

		# Total loss is a weighted sum of content and style losses
		total_loss = content_weight * content_loss + style_weight * style_loss

		# Backward pass and parameter update
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

		global_step += 1

		# verbose
		if i % print_per == 0:
			finish_training_time = time.time()
			loss_val = total_loss.item() / (height * width * channels)
			print_str = ( "Adam epoch: " + str(i) + "\t\t loss: " + str(round(loss_val, 8)) +"\t time passed: " + str(timedelta(seconds=finish_training_time - start_training_time)) )
			print(print_str)

	# -------------------------
	# Phase 2: LBFGS Optimization
	# -------------------------
	optimizer = torch.optim.LBFGS([target_image], lr=learning_rate_LBFGS)

	# Define the closure function for LBFGS:
	def closure():
		optimizer.zero_grad()
		target_features = get_features(model, target_image)

		content_loss = torch.mean( (target_features[content_layer] - content_features[content_layer]) ** 2 )

		style_loss = 0
		for style_layer in style_weights:
			target_feature = target_features[style_layer]
			target_gram = gram_matrix(target_feature)
			style_gram = style_grams[style_layer]
			current_style_loss = style_weights[style_layer] * torch.mean( (target_gram - style_gram) ** 2 )
			_, d, h, w = target_feature.shape
			style_loss += current_style_loss / (d * h * w)

		total_loss = content_weight * content_loss + style_weight * style_loss
		total_loss.backward()
		return total_loss

	# LBFGS training loop
	for i in range(1, LBFGS_epoch + 1):
		loss = optimizer.step(closure)
		optimizer.zero_grad()  # clear gradients after the optimizer step
		global_step += 1

		if i % print_per == 0:
			finish_training_time = time.time()
			loss_val = loss.item() / (height * width * channels)
			print_str = ( "LBFGS epoch: " + str(i) + "\t\t loss: " + str(round(loss_val, 8)) + "\t\t time passed: " + str(timedelta(seconds=finish_training_time - start_training_time)) )
			print(print_str)

	return target_image


# MAIN --------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Parse optional command-line arguments
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content_image_path", type=str, default=default_content_image_path,
                        help="Path to the content image")
    parser.add_argument("--style_image_path", type=str, default=default_style_image_path,
                        help="Path to the style image")
    parser.add_argument("--style_weight", type=float, default=default_style_weight,
                        help="Weight for style transfer")
    parser.add_argument("--learning_rate_Adam", type=float, default=default_learning_rate_Adam,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument("--learning_rate_LBFGS", type=float, default=default_learning_rate_LBFGS,
                        help="Learning rate for the LBFGS optimizer")
    parser.add_argument("--adam_epoch", type=int, default=default_adam_epoch,
                        help="Number of epochs for Adam optimization")
    parser.add_argument("--LBFGS_epoch", type=int, default=default_LBFGS_epoch,
                        help="Number of epochs for LBFGS optimization")
    args = parser.parse_args()

    # Override default values with parsed arguments
    content_image_path = args.content_image_path
    style_image_path = args.style_image_path
    style_weight = args.style_weight
    learning_rate_Adam = args.learning_rate_Adam
    learning_rate_LBFGS = args.learning_rate_LBFGS
    adam_epoch = args.adam_epoch
    LBFGS_epoch = args.LBFGS_epoch

	# get content and style images
    content_image = load_image(content_image_path).to(device=device)
    style_image = load_image(style_image_path, shape=content_image.shape[-2:]).to(device=device)

    model = get_model()
    target_image = train(model, content_image, style_image, adam_epoch=adam_epoch, LBFGS_epoch=LBFGS_epoch)
    save_image(target_image)

