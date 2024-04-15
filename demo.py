from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from global_hist_eq import perform_global_hist_equalization
from global_hist_eq import my_hist
from adaptive_hist_eq import perform_adaptive_hist_equalization


# Load the image
img1 = Image.open("input_img.png").convert('L')

# Convert the image to a NumPy uint8 array
img_array = np.array(img1).astype(np.uint8)

# Perfrom global equalization
equalized_img2 = perform_global_hist_equalization(img_array)
img2 = Image.fromarray(equalized_img2)

# Define the contextual region size
region_len_h = 48
region_len_w = 64

# Perform the adaptive equalization
equalized_img3 = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)
img3 = Image.fromarray(equalized_img3)

# Save resulting images locally
img1.save("img1.png")
img2.save("img2.png")
img3.save('img3.png')

# Plot histograms and save locally
my_hist(img_array, 256, 'hist1.png')
my_hist(equalized_img2, 256, 'hist2.png')
my_hist(equalized_img3, 256, 'hist3.png')