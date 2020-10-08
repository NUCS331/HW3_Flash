import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib
import cv2

savedir = 'output'

def save_fig_as_png(figtitle):
    '''
    Saves the current figure into the output folder
    The figtitle should not contain the ".png".
    This helper function shoudl be easy to use and should help you create the figures 
    needed for the report
    
    The directory where images are saved are taken from savedir in "Code.py" 
    and should be included in this function.
    
    Hint: The plt.gcf() might come in handy
    Hint 2: read about this to crop white borders
    https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    
    '''
    raise NotImplementedError

def load_imgs(folder):
    """
    This function loads your images into two numpy arrays.
    
    One for the noisy images and one for the flash images.
    
    Normalize the images. They should have float values!
    
    args:
        path (str): the path to the folder of the images, flash and no flash.
        
    outputs:
        img_noisy (np.ndarray) dtype = float32 : this is a numpy array containing both images
        img_flash (np.ndarray) dtype = float32 : this is a numpy array containing both images

    """
    raise NotImplementedError



def plot_imgs(img_noisy,img_flash):
    """
    plots the images in a subplot (left and right)
    
    make sure to have a large fontsize
    
    you can use plt.figure(figsize=(X,Y)) to make a larger figure
    so that the image doesn't appear too small
    
    
    args:
        img_noisy (np.ndarray): the noisy image
        img_flash (np.ndarray): the flash image

    outputs:
        void: plots the images!
    """
    raise NotImplementedError


def imshow_single_bilateral_filter(img_noisy,img_bilateral,xmin=600,xmax=700,ymin=400,ymax=500):
    """
    
    Visualizes the effect of the bilateral filter.
    
    The top row will show the full image (left img_noisy, right the bilateral)
    The bottom row will show a cropped/zoomed version where the input dimensions
    are defined by the function argument
    
    To visualize which part you are cropping you should overlay a rectangle
    over the region that you are corrping.
    
    You can e.g. use this resource to learn how to create a rectangle over a plot:
    https://www.delftstack.com/howto/matplotlib/how-to-draw-rectangle-on-image-in-matplotlib/
    
    
    args:
        img_noisy(np.array): noisy image
        img_bilateral(np.array): the bilateral filtered image
        xmin(int): the left start index for crop
        xmax(int): the right end index for crop
        ymin(int): the top index for crop
        ymax(int): the bottom index for crop
    
    """
    raise NotImplementedError

    
def bilateral_filter(img, sigma_r, sigma_s):
    """
    computes a bilateral filter on a rgb image.
    
    Bilateral filtering is a edge-detecting, noise-reducing, smoothing filter 
    for images. It replaces the pixels with an average of the nearby pixels, 
    which is dependent on a Gaussian kernel in the spatial domain (σ_s) and 
    also the range (intensity of pixels) domain (σ_r)
    
    this can be implemented by yourself or an external party code (easier, recommended)
    
    Note: Not every bilateral function works on RGB images. You might need to calculate
    it seperately for each channel
    
    args:
        img_color (np.ndarray): rgb image
                
        sigma_r (float): the standard deviation of the Guassian kernel in the range (intensity of pixel) 
            domain.
            
        sigma_s (float): the standard deviation of the Guassian kernel in the spatial domain. 
        
    output:
        bilteral_filter (np.ndarray): the filtered image channel, with edges more pronounced,
        noise-reduced, and smoothed.
    """
    raise NotImplementedError


def get_range_bilateral_filter():
    """
    Returns a few values for both the spatial and range parameter
    for the bilateral filter.
    
    This function doesn't do any computing. 
    You can e.g. use the following parameter, but they might not be the best
    
    range = 0.05,0.1,0.2,0.3,0.4,0.5
    spatial = 1,2,4,8,16
    
    output:
        range_of_sigma_r(1d-array)
        range_of_sigma_s(1d-array)
    """
    raise NotImplementedError


def filter_all_images_with_bilateral(img, range_of_sigma_r, range_of_sigma_s):
    """
    
    Applies the bilateral filter function (that you've implemented) 
    for each combination in range_of_sigma_r and range_of_sigma_s
    
    E.g. if len(range_of_sigma_r) = 5 and len(range_of_sigma_s) = 6
    you have to compute 30 bilateral filtered images
    
    Return the result in large 5 dimensional array.
    
    args:
        img(np.array): image to be filtered
        range_of_sigma_r(1d-array): range sigma array for bilateral filter
        range_of_sigma_s(1d-array): spatial sigma array for bilateral filter
    output: 
        filtered_imgages(np.array): filtered output with shape = (numX,numY,3,len(range),len(spatial))
    """
    # this sets up a empty numpy array to store the newly bilateral filtered images
    raise NotImplementedError



def plot_bilateral_filter(img, range_of_sigma_r, range_of_sigma_s, xMin = None, xMax = None, yMin = None, yMax = None):
    """
    plots the bilateral filtered images with different ranges of sigma_r and sigma_s.
    
    Hint: You will have to call the bilateral filter in 2 for loops that loop through both 
    range_of_sigma_r and range_of_sigma_s
    
    args:
        img (np.ndarray): image to apply the bilateral filter on.
        range_of_sigma_r (lst) dtype = float: list of the desired sigma_r to be plotted.
        range_of_sigma_s (lst) dtype = float: list of the desired sigma_s to be plotted.
        xMin (int): Min x you want to plot (used to zoom in and look at a specific region of the image)
        xMax (int): Max x you want to plot (used to zoom in and look at a specific region of the image)
        yMin (int): Min y you want to plot (used to zoom in and look at a specific region of the image)
        yMax (int): Max y you want to plot (used to zoom in and look at a specific region of the image)
    
    output:
        void, but should be subplots of the different results due to sigma_r and sigma_s.
    """

    #makes plot, and sets up the number of steps to change for sigma, and max(A), which is the max pixel value of the image.
    raise NotImplementedError

def visualize_detail_layer_single(img_flash,filtered,detail,eps,xmin=300,xmax=400,ymin=400,ymax=500):
    '''
    
    Visualizes the detail layer.
    
    This is very similair to the function "imshow_single_bilateral_filter",
    however now we have 3 images on the top and bottom.
    
    HINT: The detail image might not be scaled between 0 and 1.
    In order to see the full detail layer image you need to normalize 
    it with the maximum value to be between 0 and 1.

    
    To visualize which part you are cropping you should overlay a rectangle
    over the region that you are corrping.
    
    You can e.g. use this resource to learn how to create a rectangle over a plot:
    https://www.delftstack.com/howto/matplotlib/how-to-draw-rectangle-on-image-in-matplotlib/
    
    
    args:
        img_flash (np.ndarray): input_image
        filtered (np.ndarray): bilateral filtered image
        detail (np.array): the detail image
        eps (float): The epsilon used in the formula
        xMin (int): Min x you want to plot (used to zoom in and look at a specific region of the image)
        xMax (int): Max x you want to plot (used to zoom in and look at a specific region of the image)
        yMin (int): Min y you want to plot (used to zoom in and look at a specific region of the image)
        yMax (int): Max y you want to plot (used to zoom in and look at a specific region of the image)

    '''
    raise NotImplementedError

    
    
    
def calc_detail_layer(img,img_filtered,eps):
    """
    Take a flash and denoised flash image and calculates the detail layer 
    acoording to this function:
    
    img_detail = \frac{F+\epsilon}{F_{d}+\epsilon}
    
    
    args:
        img(np.array): RGB flash image
        img_filtered(np.array): Bilateral filtered RGB flash image
        eps: epsilon offset used in formula
    output:
        detail_layer(np.array) detail layer in RGB format
    """
    raise NotImplementedError


def calc_detail_layer_from_scratch(img,sigma_r,sigma_s,eps):
    """
    
    Calculate the detail layer from a flash_image
    using the following formula:
    
    img_detail = \frac{F+\epsilon}{F_{d}+\epsilon}
    where $\epsilon$ is a small number (e.g. 0.1 or 0.2)

    F : input flash image
    F_d: bilateral filtered image
    
    HINT: Implement the function calc_detail_layer(img,img_filtered,eps
    and call this function inside calc_detail_layer_from_scratch
    
    args:
        img(np.array): RGB image flash
        sigma_r(float): range sigma for bilateral filter
        sigma_s(float): spatial sigma for bilateral filter
        eps(float): cut-off value for filtering
    
    output:
        detail(nd.array): the filtered image
        filtered(nd.array): the bilateral filtered image
    """
    raise NotImplementedError

    
def visualize_detail_layer(img,xmin=None,xmax=None,ymin=None,ymax=None):
    """
    
    Takes in a flash image and calcualates the detail layer for those parameters
    
    You'll have to define a set of parameter for eps, sigma_r and sigma_s
    
    It probably makes sense to fix one parameter (e.g. sigma_s) to a reasonable value
    and then vary only the other parameters
    
    I suggest calculating about 16 (4x4 images), otherwise it's hard to see the effect.
    
    You'll have to go through each combination in your parameters, calculate
    the detail layer using "calc_detail_layer_from_scratch" and then display it
    
    NOTE:
    1. Give a good title of each subfigure (large fontsize and descriptive)
    2. Use for loops to go through the parameters. Don't write each subplot manually
    3. Make the figure size large enough so that you can see something
    
    The xmin,xmax,ymin,ymax are either None or integer value.
    For easy implementation checkout the following command:
    img[xmin:xmax,ymin:ymax] - What happens if xmin is None ?! Numpy is quite smart,
    you don't even need an if-statement if you do it correctly!
    
    args:
        img(np.array): The image you use to calculate the detail layer
        xmin,xmax,ymin,ymax(int): Same definitions as before
        if None show the whole image, if they are not Non you have to crop them
    
    """
    raise NotImplementedError


def visualize_fusion(img_flash,img_flash_filtered,img_noisy,img_noisy_filtered,
                     fused,detail,eps,xmin=None,xmax=None,ymin=None,ymax=None):
    """
    Visualizes the complete pipeline in a large subfigure
    
    Function arguments are the same as in the functions above
    
    xmin,xmax,... have the same purpose as before.
    
    """
    raise NotImplementedError

    
    
def fuse_flash_no_flash(img_noisy,img_flash,sigma_r,sigma_s,eps):
    """
    
    Calculates the pipeline to do the flash-no-flash photography
    
    Hint: Use the subfunction that you've already implemented in this assignment
    
    args:
        img_nosy(np.ndarray): noisy image
        
        img_flash(np.ndarray): flash image
        
        sigma_r(float): range sigma for bilateral
        sigma_s(float): spatial sigma for bilateral
        eps(float): cut off value for detail layer correction
        
    outputs:
        fused_image (np.ndarray) fused image
        detail (np.ndarray) the detail layer
        img_flash_filtered (np.ndarray)  the filtered flash image
        img_noisy_filtered (np.ndarray) the noisy image filtered
    """
    
    raise NotImplementedError



def complete_pipeline(foldername,sigma_r,sigma_s,eps):
    """
    basically does the same as fuse_flash_no_flash, so call this function inside.
    
    However, now we're not giving the images as input, but we're loading
    the image directly from the foldername.
    
    You can use your load_imgs function for this.
    
    
    args:
        foldername (str): foldername
        sigma_r(float)
        sigma_s(float)
        eps(float)
    
    outputs:
        img_nosy
        img_flash
        fused_image
        detail
        img_flash_filtered
        img_noisy_filtered
    """
    raise NotImplementedError
