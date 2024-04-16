from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from global_hist_eq import get_equalization_transform_of_img
from global_hist_eq import my_hist

def calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w):
    # Initialize the dictioany
    region_to_eq_transform = {}

    # Iterate all contextual regions and add their transformation to the dictionary
    for h in range(img_array.shape[0] // region_len_h):
        h1 = h * region_len_h
        h2 = (h + 1) * region_len_h
        for w in range(img_array.shape[1] // region_len_w):
            w1 = w * region_len_w
            w2 = (w + 1) * region_len_w
            region_to_eq_transform[(h, w)] = get_equalization_transform_of_img(img_array[h1:h2, w1:w2])
    
    return region_to_eq_transform

def perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w):
    # Get image size, initialize equalized image and calculate number of contextual regions
    m, n = img_array.shape
    equalized_img = np.zeros((m, n)).astype(np.uint8)
    h_range = m // region_len_h
    w_range = n // region_len_w

    # Retrive the dictionary connecting every contexual region to the corresponding equalization transformation 
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    # Fill inner contextual regions by iterating all the pixels and interpolating all the neighboring transformations
    for hp in range(region_len_h, (h_range - 1) * region_len_h):
        for wp in range(region_len_w, (w_range - 1) * region_len_w):
            # Find nearest region center
            hc = (hp // region_len_h) * region_len_h - region_len_h // 2 
            wc = (wp // region_len_w) * region_len_w - region_len_w // 2
            if hp - hc >= region_len_h: hc += region_len_h
            if wp - wc > region_len_w: wc += region_len_w

            a = (wp - wc) / region_len_w
            b = (hp - hc) / region_len_h
            
            # Find corresponing coordinates for dictinary retrieval
            h = hc // region_len_h
            w = wc // region_len_w
            if hp == 90  and wp == 120:
                print(h,w)
            Tmm = region_to_eq_transform[(h, w)]
            Tmp = region_to_eq_transform[(h, w + 1)]
            Tpm = region_to_eq_transform[(h + 1, w)]
            Tpp = region_to_eq_transform[(h + 1, w + 1)]

            equalized_img[hp, wp] = (1 - a) * (1 - b) * Tmm[img_array[hp, wp]]
            equalized_img[hp, wp] += (1 - a) * b * Tpm[img_array[hp, wp]] 
            equalized_img[hp, wp] += a * (1 - b) * Tmp[img_array[hp, wp]]
            equalized_img[hp, wp] += a * b * Tpp[img_array[hp, wp]]
    
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
    
    # Return image cropped accordingly
    return equalized_img[0 : m - m%region_len_h, 0 : n - n%region_len_w]

# Example usage

# Load the image
#img1 = Image.open("input_img.png").convert('L')

# Convert the image to a NumPy uint8 array
#img_array = np.array(img1).astype(np.uint8)

# Define the contextual region size
#region_len_h = 48
#region_len_w = 64

# Perform the adaptive equalization
#equalized_img = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)

# Save results to a different image in the same directory
#img3 = Image.fromarray(equalized_img)
#img3.save('img3.png')