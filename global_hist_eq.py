from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_equalization_transform_of_img(img_array): 
    # Get image dimensions
    n1, n2 = img_array.shape
    # Define unique values number based on uint8 dtype
    L = 256
    
    # Turn array to 1d for ease of use
    img_flat = img_array.flatten()
    
    # Find the probability of each value happening
    _, counts = np.unique(img_flat, return_counts=True)
    p = counts / (n1 * n2)
    
    # Find the u vector as the cumulative sum
    u = np.cumsum(p)
    
    equalization_transform = np.round((u - u[0]) * (L - 1) / (1 - u[0]))
    return equalization_transform

def perform_global_hist_equalization(img_array):
    T = get_equalization_transform_of_img(img_array)
    equalized_img = T[img_array]
    return equalized_img.astype(np.uint8)
    
# Load the image
img1 = Image.open("input_img.png").convert('L')
# Convert the image to a NumPy uint8 array
img_array = np.array(img1).astype(np.uint8)

# Perform equalization
equalized_img = perform_global_hist_equalization(img_array)

# Save results locally
img2 = Image.fromarray(equalized_img)
img1.save("img1.png")
img2.save("img2.png")

#plt.hist(np.concatenate(equalized_img))
#plt.show()