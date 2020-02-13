#!/usr/bin/env python
# coding: utf-8

# __Submitted by__: M. Hasnain Naeem (212728) from BSCS-7B, NUST 
# 
# # Digital Image Processing 
# ## Lab 4 - Transformation Functions and Geometric Transforms from Scratch and using OpenCV
# __Objectives:__
# The objectives of this lab are: 
#     - To apply Log transformation on images and visualize results. 
#     - To apply power-law transform to correct gamma in images. 
#     - To apply various geometric transformations (like translation, rotation, scaling, sheering and affine) on images and see their effects.

# In[1]:


import os
import math
import numpy as np
import cv2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def get_gray_image(name, img_dir="files/imgs"):
    # open file and convert to grey scale
    filename = os.path.join(os.curdir, img_dir, name)
    
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return gray_img


# In[3]:


def get_intensity_counts(gray_img):
    intensity_count = np.zeros(256, "int")
    for i, row in enumerate(gray_img):
        for j, intensity_val in enumerate(row):
            intensity_count[intensity_val] += 1
    return intensity_count


# In[4]:


def plot_intensity_hists(bins, intensity_counts, saving_dir, filename):
    save_loc = os.path.join(saving_dir, filename+"_intensity_histogram"+".jpg")

    plt.bar(bins, intensity_counts, color='#0504aa')
    
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity Value')
    plt.ylabel('Intensity Value Count')
    plt.title("Intensity Value Histogram for "+filename)
    
    plt.savefig(save_loc, bbox_inches="tight", dpi=100)
    plt.show()


# In[5]:


# get all the filenames in the directory
imgs_dir = os.path.join(os.curdir, "files", "imgs")
img_names = os.listdir("files/imgs")

# get filename --> gray_image mapping for future usage
gray_imgs = dict()
for img_name in img_names:
    gray_img = get_gray_image(img_name)
    gray_imgs[img_name] = gray_img
    plt.imshow(gray_img, cmap="gray")
    plt.show()


# ### Task 1
# 1. Use python notebook and opencv’s imread function and load an image (e.g. dark.tif) 
# 2. Visualize histogram by plotting the histogram. 
# 3. Apply log transform on image and visualize the output histogram. 
# 4. Try to experiment by changing the scaling constant c 
# 5. Summarize your findings by providing a figure containing 4 plots (input image, histogram of input image, output image and histogram of output image) 

# #### Log Transformation

# In[6]:


def log_transform(img, args):
    log_transformed_img = np.zeros(img.shape, "int")

    # calculate scaling constant
    c = (255)/(math.log(1+np.max(img)))

    # transform all intensities
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            # transform current intensity value
            log_transformed_img[i, j] = c * math.log(1+val)
            
    return log_transformed_img


# In[7]:


def transform_and_hists(img_name, gray_img, trans_func, saving_dir, **kwargs):  
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
    transformed_img = trans_func(gray_img, kwargs)
    
    # new file name & location
    new_path = os.path.join(saving_dir, "transformed", img_name+"_transformed.jpg")
    plt.imsave(new_path, transformed_img, cmap="gray")
    
    print("Transformed Image:")
    plt.imshow(transformed_img, cmap="gray")
    plt.show()
    
    # plot histogram of transformed image
    intensity_count = get_intensity_counts(transformed_img)
    plot_intensity_hists(bins, intensity_count, saving_dir, img_name+"_"+trans_func.__name__+"_transformed.jpg")
    


# #### Draw Histogram of Image, Transform, Draw Histogram of Transformed Image

# In[8]:


task1_dir = os.path.join(os.curdir, "files", "task1")
task1_transformed_dir = os.path.join(task1_dir, "transformed")

# create directories doesn't exist
if not os.path.exists(task1_dir):
    os.makedirs(task1_dir)
if not os.path.exists(task1_transformed_dir):
    os.makedirs(task1_transformed_dir)

# perform operations on all the images
for img_name, gray_img in gray_imgs.items():
    transform_and_hists(img_name, gray_img, log_transform, task1_dir)


# ### Task 2
# - Read “aerial.tif” image in python notebook and store that in “img” variable. - Read the documentation of Interact from ipywidgets and create a slider named gamma. 
# - By using the value of gamma, apply a power-law transform on image and figure out which value of gamma can result in a contrast enhanced output. 
# - __HINT:__ since you have to apply this transformation to each pixel, nested for loops including the power-law operation should do the trick. 
# - Summarize your findings by providing a figure containing 4 plots (input image, histogram of input image, output image and histogram of output image) 

# #### Functions for Power Law Transformation & Histogram Generation

# In[9]:


def power_law(img, args):
    gamma_val = args["gamma_val"]
    c = args["c_val"]
    
    gamma_transformed_img = np.zeros(img.shape, "int")
    
    # transform all intensities
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            # transform current intensity value
            gamma_transformed_img[i, j] = int(c * math.pow(val, gamma_val))
            
    return gamma_transformed_img


# In[10]:


def power_law_pixel(intensity_val, args):
    gamma_val = args["gamma_val"]    
    c = args["c_val"]

    transformed_intensity = int(c * math.pow(intensity_val, gamma_val))
            
    return transformed_intensity


# In[11]:


def gen_power_law_table(args):
    # generate gamma table
    gamma_table = dict()
    for i in range(256):
        gamma_table[i] = power_law_pixel(i, args)
        
    return gamma_table


# In[12]:


def power_law_using_table(img, args):
    gamma_table = args["power_law_table"]
    gamma_transformed_img = np.zeros(img.shape, "int")
    
    # transform all intensities
    for i, row in enumerate(img):
        for j, val in enumerate(row):
            # transform current intensity value
            gamma_transformed_img[i, j] = gamma_table[val]
            
    return gamma_transformed_img


# #### Make the required directories & open the file

# In[13]:


task2_dir = os.path.join(os.curdir, "files", "task2")
task2_transformed_dir = os.path.join(task2_dir, "transformed")

# create directories doesn't exist
if not os.path.exists(task2_dir):
    os.makedirs(task2_dir)
if not os.path.exists(task2_transformed_dir):
    os.makedirs(task2_transformed_dir)

# transform using Power Law
task2_img_name = "aerial.tif"
task2_img = gray_imgs[task2_img_name]


# #### Power Law Transformation with Gamma Value & Scaling Value Sliders

# ##### Without using Transformation Table

# In[54]:


from ipywidgets import interact

def gamma_slider(gamma_val, c_val):
    args = {"gamma_val": gamma_val, "c_val": c_val}
    transformed_img = power_law(task2_img, args)
    plt.imshow(transformed_img, cmap="gray")

# gamma value & scaling factor sliders
interact(gamma_slider, gamma_val=(0.0, 2.0), c_val=(0.0, 2.0));

# ignore below lines
# for lab submission, load and show the slider output screenshot; because slider won't appear in HTML version of notebook
plt.figure(figsize = (7,7))
screenshot_1 = cv2.imread("screenshot_1.PNG")
plt.imshow(screenshot_1)
plt.show()


# ##### Using Transformation Table

# In[55]:


from ipywidgets import interact

def gamma_slider(gamma_val, c_val):
    args = {"gamma_val": gamma_val, "c_val": c_val}
    args["power_law_table"] = gen_power_law_table(args)
    transformed_img = power_law_using_table(task2_img, args)
    plt.imshow(transformed_img, cmap="gray")

# gamma value & scaling factor sliders
interact(gamma_slider, gamma_val=(0.0, 2.0), c_val=(0.0, 2.0));


# ignore below lines
# for lab submission, load and show the slider output screenshot; because slider won't appear in HTML version of notebook
plt.figure(figsize = (7,7))
screenshot_2 = cv2.imread("screenshot_2.PNG")
plt.imshow(screenshot_1)
plt.show()


# #### Histogram Comparisons

# #### Without using Transformation Table

# In[16]:


transform_and_hists(task2_img_name, task2_img, power_law, task2_dir, gamma_val=0.4, c_val=1)


# ##### Using Transformation Table

# In[17]:


args = {"gamma_val": 0.4, "c_val":1}
power_law_table = gen_power_law_table(args)
transform_and_hists(task2_img_name, task2_img, power_law_using_table, task2_dir, gamma_val=0.4, c_val=1, power_law_table=power_law_table)


# ### Task 3
# Read “messi5.jpg” image and apply following transformations using cv2.warpaffine() function: 
# - Translation 
# - Rotation
# - Sheering in x 
# - Sheering in y 
# - Random affine transform 
# 
# Summarize your findings by providing a figure which contains images before and after geometric transformation. 
#  

# In[18]:


# open files
task3_img_name = "messi5.jpg"
task3_img = gray_imgs[task3_img_name]
plt.imshow(task3_img, cmap="gray")
plt.show()


# #### Translation

# In[19]:


trans_x = 100
trans_y = 100

task3_img_rows, task3_img_cols =task3_img.shape
img3_trans_shape = (task3_img_cols+100, task3_img_rows+100)

trans_M = np.float32([[1,0,trans_x],[0,1,trans_y]])

# enlarged image to contain the translated image
task3_img_translated = np.zeros((task3_img.shape[0]+100, task3_img.shape[1]+100))
# paste the image into the container image
task3_img_translated[:task3_img.shape[0],:task3_img.shape[1]] = task3_img
plt.imshow(task3_img_translated, cmap="gray")
plt.show()
# transform and show
translated_img = cv2.warpAffine(task3_img, trans_M, img3_trans_shape)
plt.imshow(translated_img, cmap="gray")
plt.show()


# #### Rotation

# In[20]:


rot_angle = 90
rotation_M = cv2.getRotationMatrix2D((img3_trans_shape[0]/2,img3_trans_shape[1]/2),rot_angle,1)

# enlarged image to contain the translated image
task3_img_rotated = np.zeros((task3_img.shape[0]+100, task3_img.shape[1]+100))
# paste the image into the container image
task3_img_rotated[:task3_img.shape[0],:task3_img.shape[1]] = task3_img
plt.imshow(task3_img_rotated, cmap="gray")
plt.show()
# transform and show
rotated_img = cv2.warpAffine(task3_img_rotated, rotation_M, img3_trans_shape)
plt.imshow(rotated_img, cmap="gray")
plt.show()


# #### Sheering in X

# In[21]:


shear_x = .1
shear_x_M = np.float32([[1,shear_x,0],[0,1,0]])

# enlarged image to contain the translated image
task3_img_shear_x = np.zeros((task3_img.shape[0]+100, task3_img.shape[1]+100))
# paste the image into the container image
task3_img_shear_x[:task3_img.shape[0],:task3_img.shape[1]] = task3_img
plt.imshow(task3_img_shear_x, cmap="gray")
plt.show()
# transform and show
shear_x_img = cv2.warpAffine(task3_img_shear_x, shear_x_M, img3_trans_shape)
plt.imshow(shear_x_img, cmap="gray")
plt.show()


# #### Sheering in Y

# In[22]:


shear_y = .1
shear_y_M = np.float32([[1,0,0],[shear_y,1,0]])

# enlarged image to contain the translated image
task3_img_shear_y = np.zeros((task3_img.shape[0]+100, task3_img.shape[1]+100))
# paste the image into the container image
task3_img_shear_y[:task3_img.shape[0],:task3_img.shape[1]] = task3_img
plt.imshow(task3_img_shear_y, cmap="gray")
plt.show()
# transform and show
shear_y_img = cv2.warpAffine(task3_img_shear_y, shear_y_M, img3_trans_shape)
plt.imshow(shear_y_img, cmap="gray")
plt.show()


# #### Random Affine Transformation

# In[23]:


pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

# enlarged image to contain the translated image
task3_img_random = np.zeros((task3_img.shape[0]+100, task3_img.shape[1]+100))
# paste the image into the container image
task3_img_random[:task3_img.shape[0],:task3_img.shape[1]] = task3_img
plt.imshow(task3_img_random, cmap="gray")
plt.show()

# transform and show
random_M = cv2.getAffineTransform(pts1,pts2)
random_trans_img = cv2.warpAffine(task3_img_random,random_M,img3_trans_shape)
plt.imshow(random_trans_img, cmap="gray")
plt.show()


# ### Task 4
# You may have noticed that in task 2 when using interact with some real-time processing, the system takes some time to process. 
# - This is usually caused by applying exponential mathematical operation withing nested for loops. 
# - You can use the concept of mapping to apply the transformation function in an efficient manner! 
# - Try out both techniques (i.e. power-law in nested for loops and mapping) by increasing the size of input image and calculate the time take by each. 
# - Conclude your findings in the form of graph in which x-axis should indicate the increasing size of image and y-axis should indicate time taken for processing. 
#  

# #### Trying Power Law Transformation with & without Transformation Table for Processing Time Comparison

# In[35]:


# upscale the task 3 image for task 4
print('Original Dimensions : ', task3_img.shape)
 
# increase size to 2X
scale_percent = 1000 # percent of original size
task4_img_width = int(task3_img.shape[1] * scale_percent / 100)
task4_img_height = int(task3_img.shape[0] * scale_percent / 100)
task4_img_dim = (task4_img_width, task4_img_height)

task4_img = cv2.resize(task3_img, task4_img_dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ', task4_img.shape)

plt.imshow(task4_img, cmap="gray")
plt.show()


# In[38]:


import time


# In[43]:


start_time = time.time()

args = {"gamma_val": 0.4, "c_val":1}
power_law_table = gen_power_law_table(args)
args["power_law_table"] = power_law_table
task4_img_with_table = power_law_using_table(task4_img, args)

print("--- %s seconds ---" % (time.time() - start_time))


# In[40]:


plt.imshow(task4_img_with_table, cmap="gray")
plt.show()


# In[41]:


start_time = time.time()

args = {"gamma_val": 0.4, "c_val":1}
task4_img_without_table = power_law(task4_img, args)

print("--- %s seconds ---" % (time.time() - start_time))


# In[42]:


plt.imshow(task4_img_without_table, cmap="gray")
plt.show()


# #### Comparison Results
# - Gray version of "Messi5.jpg" was upscaled to (2800, 4500) (10 times) for comparison of Power Law using Transformation Table and without the Transformation Table.
# - It took __22.32 seconds when transformation table was not used__.
# - It took __8 seconds when transformation table was used__.

# In[ ]:




