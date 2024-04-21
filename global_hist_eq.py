import numpy as np

def get_equalization_transform_of_img(img_array):

#Function that calculates the equaliation transfrom of an image
#  Input:
#   img_array: a numpy array of a uint8 grayscale image
#  Output:
#   equalization_transform: the equlization tranformation
    
    # Define the number of unique values based on uint8 dtype
    L = 256

    # Initialize the output
    equalization_transform = np.zeros(L)   
     
    # Flatten array for easy access
    img_flat = img_array.flatten()    
    
    # Find the probability of each value happening
    p = np.bincount(img_flat, minlength=L) / len(img_flat)

    # Find the u vector as the cumulative sum
    u = np.cumsum(p)
    
    # Calculate the equalization transform
    equalization_transform = np.round(((u - u[0]) / (1 - u[0])) * (L - 1))
    return equalization_transform

def perform_global_hist_equalization(img_array):
# Function that performs the global equalization technique to an image
#  Input:
#   img_array: a numpy array of a uint8 grayscale image
#  Output: The equalized image as a uint8 numpy array
    
    T = get_equalization_transform_of_img(img_array)
    return T[img_array].astype(np.uint8)

# Example Usage

# Load the image
#img1 = Image.open("input_img.png").convert('L')
# Convert the image to a NumPy uint8 array
#img_array = np.array(img1).astype(np.uint8)

# Perform global equalization
#equalized_img = perform_global_hist_equalization(img_array)

# Save results locally
#img2 = Image.fromarray(equalized_img)
#img1.save("img1.png")
#img2.save("img2.png")

# Show histogram
#plt.hist(np.concatenate(img_array))
#plt.show()
#plt.hist(np.concatenate(equalized_img))
#plt.show()