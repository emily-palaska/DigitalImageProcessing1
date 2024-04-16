#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def my_hist(data, bins=256, filename='hist.png'):
    # Make array 1d
    data_flat = data.flatten()

    # Find minimum and maximum values in the data
    min_val = np.min(data_flat)
    max_val = np.max(data_flat)

    # Calculate bin width
    bin_width = (max_val - min_val) / bins

    # Initialize histogram counts
    hist_counts = np.zeros(bins, dtype=int)

    # Iterate through data and count occurrences in each bin
    for value in data_flat:
        bin_index = int((value - min_val) // bin_width)
        if bin_index == bins:  # Handle values equal to max_val
            bin_index -= 1
        hist_counts[bin_index] += 1

    # Plot and save histogram
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    plt.bar(bin_edges[:-1], hist_counts, width=bin_width, edgecolor='blue')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.savefig(filename)
    plt.close()

def get_equalization_transform_of_img(img_array): 
    # Get image dimensions
    n1, n2 = img_array.shape
    # Define unique values number based on uint8 dtype
    L = 256
    equalization_transform = np.zeros(L)

    # Turn array to 1d for ease of use
    img_flat = img_array.flatten()
    
    # Find the probability of each value happening
    p = np.bincount(img_flat, minlength=256) / (n1 * n2)

    # Find the u vector as the cumulative sum
    u = np.cumsum(p)
    
    # Calculate the equalization transform
    equalization_transform = np.round((u - u[0]) * (L - 1) / (1 - u[0]))
    return equalization_transform

def perform_global_hist_equalization(img_array):
    T = get_equalization_transform_of_img(img_array)
    equalized_img = T[img_array]
    return equalized_img.astype(np.uint8)

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