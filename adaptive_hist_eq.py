from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from global_hist_eq import get_equalization_transform_of_img

def calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w):
    # Initialize the dictioany
    region_to_eq_transform = {}

    # Iterate all contextual regions and add their transformation to the dictionary
    for h in range(round(img_array.shape[0] / region_len_h)):
        h1 = h * region_len_h
        h2 = (h + 1) * region_len_h
        for w in range(round(img_array.shape[1] / region_len_w)):
            w1 = w * region_len_w
            w2 = (w + 1) * region_len_w
            region_to_eq_transform[(h, w)] = get_equalization_transform_of_img(img_array[h1:h2, w1:w2])
    
    return region_to_eq_transform

def perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w):
    # Get image size and initialize equalized image
    m, n = img_array.shape
    equalized_img = np.zeros((m,n)).astype(np.uint8)
    
    # Retrive the dictionary connecting every contexual region to the corresponding equalization transformation 
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    h_range = round(img_array.shape[0] / region_len_h)
    w_range = round(img_array.shape[1] / region_len_w)

    # Fill outer contextual regions without performing interpolation
    for h in range(h_range):
        h1 = h * region_len_h
        h2 = (h + 1) * region_len_h
        for w in range(w_range):
            w1 = w * region_len_w
            w2 = (w + 1) * region_len_w
            if h == 0 or w == 0 or h == h_range - 1 or w == w_range - 1:
                T = region_to_eq_transform[(h, w)]
                equalized_img[h1:h2, w1:w2] = T[img_array[h1:h2, w1:w2]]
    
    # Fill inner contextual regions by interpolating the 4 neighboring transformations


    for hp in range(region_len_h, m - region_len_h):
        for wp in range(region_len_w, n - region_len_w):     
            hm = (hp // region_len_h) * region_len_h
            wm = (wp // region_len_w) * region_len_w
            a = (wp - wm) / region_len_w
            b = (hp - hm) / region_len_h
            h = hm // region_len_h
            w = wm // region_len_w
            
            Tmm = region_to_eq_transform[(h, w)]
            Tmp = region_to_eq_transform[(h, w + 1)]
            Tpm = region_to_eq_transform[(h + 1, w)]
            Tpp = region_to_eq_transform[(h + 1, w + 1)]

            equalized_img[hp, wp] = a * b * Tmm[img_array[hp, wp]]
            equalized_img[hp, wp] += a * (1 - b) * Tpm[img_array[hp, wp]]
            equalized_img[hp, wp] += (1 - a) * b * Tmp[img_array[hp, wp]]
            equalized_img[hp, wp] += (1 - a) * (1 - b) * Tpp[img_array[hp, wp]]

    return equalized_img



# Example usage

# Load the image
img1 = Image.open("input_img.png").convert('L')
# Convert the image to a NumPy uint8 array
img_array = np.array(img1).astype(np.uint8)

# Get size and region size
m,n = img_array.shape
region_len_h = 36
region_len_w = 48

# Perform equalization
equalized_img = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)

# Save results
img3 = Image.fromarray(equalized_img)
img3.save('img3.png')