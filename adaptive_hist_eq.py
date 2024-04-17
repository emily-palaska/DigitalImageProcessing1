import numpy as np
from global_hist_eq import get_equalization_transform_of_img

def calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w):
# Function that creates a dictinary of transformations for every contextual region of an image
#  Input:
#   img_array: a numpy array of a uint8 grayscale image
#   region_len_h: the region length for the first dimension
#   region_len_w: the region length for the second dimension
#  Output:
#   region_to_eq_tranform: a dictinary that matches every contextual region to its equalization transformation
    
    # Initialize the output
    region_to_eq_transform = {}
    
    # Get the image size and calculate range of iteration
    m, n = img_array.shape
    h_range = m // region_len_h
    w_range = n // region_len_w

    # Iterate all contextual regions and add their transformation to the dictionary
    # If the image is not perfectly dividable by the region lengths, then the last contextual regions of
    # each dimension get the remaining pixels
    for h in range(h_range):
        h1 = h * region_len_h
        h2 = m - 1 if h == h_range - 1 else (h + 1) * region_len_h
        for w in range(w_range):
            w1 = w * region_len_w
            w2 = n - 1 if w == w_range - 1 else (w + 1) * region_len_w
            region_to_eq_transform[(h, w)] = get_equalization_transform_of_img(img_array[h1:h2, w1:w2])
    return region_to_eq_transform

def perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w):
# Function that perfroms the adaptive equalization technique to an image
#  Input:
#   img_array: a numpy array of a uint8 grayscale image
#   region_len_h: the region length for the first dimension
#   region_len_w: the region length for the second dimension
#  Output:
#    equalized_img: the resulting image as in uint8 numpy array
    
    # Initialize the output
    equalized_img = img_array.copy()
    
    # Get the image size and calculate number (range) of contextual regions
    m, n = img_array.shape
    h_range = m // region_len_h
    w_range = n // region_len_w

    # Retrieve the dictionary connecting every contexual region to the corresponding equalization transformation
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

     # Fill outer contextual regions without performing interpolation
    for h in range(h_range):
        h1 = h * region_len_h
        h2 = m - 1 if h == h_range - 1 else (h + 1) * region_len_h
        for w in range(w_range):
            w1 = w * region_len_w
            w2 = n - 1 if w == w_range - 1 else (w + 1) * region_len_w
            if h == 0 or w == 0 or h == h_range - 1 or w == w_range - 1:
                T = region_to_eq_transform[(h, w)]
                equalized_img[h1:h2, w1:w2] = T[img_array[h1:h2, w1:w2]]
    
    # Get the region lengths halves for later computations            
    region_len_h_half = region_len_h // 2
    region_len_w_half = region_len_w // 2
    
    # Fill inner contextual regions by iterating all the pixels and interpolating all the neighboring transformations
    for hp in range(region_len_h_half, h_range * region_len_h - region_len_h_half):
        for wp in range(region_len_w // 2, w_range * region_len_w - region_len_w // 2):
            # Find nearest region center
            hc = (hp // region_len_h_half) * region_len_h_half
            wc = (wp // region_len_w_half) * region_len_w_half
            if hc % region_len_h == 0: hc -= region_len_h_half
            if wc % region_len_w == 0: wc -= region_len_w_half

            # Calculate the interpolation prameters
            a = (wp - wc) / region_len_w
            b = (hp - hc) / region_len_h
            
            # Find corresponing coordinates for dictionary retrieval
            h = hc // region_len_h
            w = wc // region_len_w

            # Get the corresponding transformations
            Tmm = region_to_eq_transform[(h, w)]
            Tmp = region_to_eq_transform[(h, w + 1)]
            Tpm = region_to_eq_transform[(h + 1, w)]
            Tpp = region_to_eq_transform[(h + 1, w + 1)]

            # Perfrom the interpolation with auxiliary buffer
            buffer = (1 - a) * (1 - b) * Tmm[img_array[hp, wp]]
            buffer += (1 - a) * b * Tpm[img_array[hp, wp]]
            buffer += a * (1 - b) * Tmp[img_array[hp, wp]]
            buffer += a * b * Tpp[img_array[hp, wp]]
            equalized_img[hp, wp] = buffer
    return equalized_img

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