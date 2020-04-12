#!/usr/bin/env python
# coding: utf-8

# __Submitted by__: M. Hasnain Naeem (212728) from BSCS-7B, NUST 
# 
# # Digital Image Processing 
# ## Assignment 2 - Image Morphology

# In[44]:


import os
import math
import numpy as np
import cv2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# __Utility Functions for Common Steps__

# In[3]:


def get_gray_image(name, img_dir="files/imgs"):
    """
        opens file and returns binary version
    """ 
    filename = os.path.join(os.curdir, img_dir, name)
    
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return gray_img


# In[4]:


def get_intensity_counts(gray_img):
    """
        calculates count for each intensity value which can be used for 
        histogram 
    """ 
    intensity_count = np.zeros(256, "int")
    for i, row in enumerate(gray_img):
        for j, intensity_val in enumerate(row):
            intensity_count[intensity_val] += 1
    return intensity_count


# In[5]:


def plot_intensity_hists(bins, intensity_counts, saving_dir, filename):
    """
        Plots the histogram according to given counts
        Also, saves the histogram to specificed directory
    """
    save_loc = os.path.join(saving_dir, filename+"_intensity_histogram"+".jpg")

    plt.bar(bins, intensity_counts, color='#0504aa')
    
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity Value')
    plt.ylabel('Intensity Value Count')
    plt.title("Intensity Value Histogram for "+filename)
    
    plt.savefig(save_loc, bbox_inches="tight", dpi=100)
    plt.show()


# In[151]:


def transform_and_hists(img_name, gray_img, trans_func, saving_dir, **kwargs):  
    """
        Draws and saves histogram of input image. Transforms and saves the image
        according to given transformation function. Also, draws and saves histogram 
        for the transformed image.
        
        Saved File Names:
            Transformed Image:  name of Transformation Function is appended at the
                                end of file name
        params:
            trans_func: Transformation Function
            **kwargs: Extra arguments to be passed to Transformation Function
    """
    
    print(img_name)
    print("************")
    
    print("Original Image:")
    plt.imshow(gray_img, cmap="gray")
    plt.show()
    
    bins = [i for i in range(256)]
    # plot intensity histogram
    intensity_count = get_intensity_counts(gray_img)
    plot_intensity_hists(bins, intensity_count, saving_dir, img_name)
    
    # transformation
    if kwargs:
        transformed_img = trans_func(gray_img, kwargs)
    else:
        transformed_img = trans_func(gray_img)
        
    # new file name & location
    new_path = os.path.join(saving_dir, "transformed", img_name+"_"+trans_func.__name__+".jpg")
    plt.imsave(new_path, transformed_img, cmap="gray")
    
    print("Transformed Image:")
    plt.imshow(transformed_img, cmap="gray")
    plt.show()
    
    # plot histogram of transformed image
    intensity_count = get_intensity_counts(transformed_img)
    plot_intensity_hists(bins, intensity_count, saving_dir, img_name+"_"+trans_func.__name__+".jpg")
    return transformed_img


# __Get key value pairs (image_name, img) of all the images in the given directory__

# In[6]:


# get all the filenames in the directory
imgs_dir = os.path.join(os.curdir, "files", "imgs")
img_names = os.listdir("files/imgs")

# get filename --> gray_image mapping for future usage
gray_imgs = dict()
for img_name in img_names:
    gray_img = get_gray_image(img_name)
    gray_imgs[img_name] = gray_img
    plt.imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    plt.show()


# ### Task 1
# First, apply binary erosion and dilation. Later analyze the difference with respect to erosion and dilation.

# In[137]:


def erosion(img, args={"dims": (5,5)}):
    kernel = np.ones((args["dims"]), np.uint8)
    eroded_img = cv2.erode(img,kernel,iterations = 1)
    
    return eroded_img


# In[138]:


def dilate(img, args={"dims": (5,5)}):
    kernel = np.ones((args["dims"]), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations = 1)
    
    return dilated_img


# #### Open Task 1 Images

# In[152]:


# open task 1 files and create directories for the transformed images
task1_dir = os.path.join(os.curdir, "files", "task1")
task1_transformed_dir = os.path.join(task1_dir, "transformed")

# create directories doesn't exist
if not os.path.exists(task1_dir):
    os.makedirs(task1_dir)
if not os.path.exists(task1_transformed_dir):
    os.makedirs(task1_transformed_dir)

# show task 1 image
task1_img_name = "english.png"
task1_img = gray_imgs[task1_img_name]
plt.imshow(task1_img, cmap="gray")
plt.show()

# plot intensity histogram
bins = [i for i in range(256)]
intensity_count = get_intensity_counts(task1_img)
plot_intensity_hists(bins, intensity_count, saving_dir=task1_dir, filename=task1_img_name)


# #### Apply Erosion

# In[153]:


task1_eroded_img = transform_and_hists(task1_img_name, task1_img, erosion, task1_dir, dims=(5,5))


# #### Apply Dilation

# In[154]:


task1_dilated_img = transform_and_hists(task1_img_name, task1_img, dilate, task1_dir, dims=(5, 5))


# ### Task 2

# Segment the foreground (Urdu characters) in the following image from its background.

# In[110]:


# open task 2 files and create directories for the transformed images
task2_dir = os.path.join(os.curdir, "files", "task2")
task2_transformed_dir = os.path.join(task2_dir, "transformed")

# create directories doesn't exist
if not os.path.exists(task2_dir):
    os.makedirs(task2_dir) 
if not os.path.exists(task2_transformed_dir):
    os.makedirs(task2_transformed_dir)

# show task 2 image
task2_img_names = ["urdu.png"]
task2_imgs ={img_name:gray_imgs[img_name] for img_name in task2_img_names}

for task2_img_name, task2_img in task2_imgs.items():
    plt.imshow(task2_img, cmap="gray")
    plt.show()
    # plot intensity histogram
    bins = [i for i in range(256)]
    intensity_count = get_intensity_counts(task2_img)
    plot_intensity_hists(bins, intensity_count, saving_dir=task2_dir, filename=task2_img_name)


# #### Segmenting the Signature

# In[156]:


def segment_sign(img, args={"rect_dims": (4, 1), "erode_dims": (5,5), "dilate_dims":(5,5)}):
    # Rectangular Kernel to be used as Structural Element
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, args["rect_dims"])
    # apply the structural element filter
    res = cv2.filter2D(img, -1, rect_kernel) 

    print("Image after structural element application:")
    plt.imshow(res, cmap="gray")
    plt.show()

    # kernel for dilation and erotion
    # Note: erotion and dilation roles are apparenly switched because we have
    #       dark foreground and light background which is opposite of usual
    erode_kernel = np.ones(args["erode_dims"])
    dilate_kernel = np.ones(args["dilate_dims"])
    
    # erode to fill the holes; but expansion of edges also occurs
    eroded_img = cv2.erode(res, erode_kernel, iterations = 2)
    # apply dilate to remove trim the edges
    dilated_img = cv2.dilate(eroded_img, dilate_kernel, iterations = 1)
    
    return dilated_img


# In[160]:


segmented_img = transform_and_hists("urdu.png", task2_imgs["urdu.png"], segment_sign, task2_dir)


# ### Directory Structure
# __Directory structure of the saved files__

# In[123]:


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


# In[124]:


list_files(os.getcwd())


# In[ ]:




