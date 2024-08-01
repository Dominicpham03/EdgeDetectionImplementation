import cv2
import numpy
import math

def load_image(file_name):
    image = cv2.imread(file_name, 0)
    image = cv2.resize(image, (800, 1200))
    return image

def display_image(image):
    window_name = 'Display window'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 1200)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.DestroyAllWindows()

def generate_gaussian(sigma, filter_w, filter_h):
    mu = 0
    pi = math.pi
    two_sigma_squared = 2*(sigma**2)

    if(filter_w == 1 or filter_h == 1):
        filter_len = 0
        if filter_w >= filter_h:
            filter_len = filter_w
        else:
            filter_len = filter_h

        center = filter_len // 2
        result = numpy.zeros(filter_len)
        gaussian_value = 1 / (math.sqrt(2 * pi) * sigma)

        for i in range(filter_len):
            dist = (i - center)
            result[i] = gaussian_value * math.exp(-((dist-mu)**2) / two_sigma_squared)
        
        return result
    
    else:
        center_w = filter_w // 2
        center_h = filter_h // 2
        result = numpy.zeros((filter_h, filter_w))
        gaussian_value = 1 / (two_sigma_squared*pi)

        for i in range(filter_h):
            for j in range(filter_w):
                dist_x = (j - center_w)**2
                dist_y = (i - center_h)**2
                result[i][j] = gaussian_value * math.exp(-((dist_x + dist_y)) / two_sigma_squared)
    
        result /= numpy.sum(result)

        return result

def apply_filter(image, filter, pad_pixels, pad_value):
    image_h, image_w = image.shape
    output = numpy.zeros_like(image)

    if pad_value == 0:
        mode = 'constant'
    else:
        mode = 'edge'

    if filter.ndim == 1:
        pad_image = numpy.pad(image, ((0,0), (pad_pixels, pad_pixels)), mode)
        filter_len = filter.shape[0]

        for i in range(image_h):
            for j in range(image_w-filter_len + 1):
                area = pad_image[i, j:j + filter_len]
                output[i,j + pad_pixels] = numpy.sum(area * filter)
            
    else:
        filter_h, filter_w = filter.shape

        pad_image = numpy.pad(image, ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels)), mode)

        for i in range(image_h):
            for j in range(image_w):
                area = pad_image[i:i + filter_h, j:j + filter_w]
                if area.shape == filter.shape:
                    output[i,j] = numpy.sum(area * filter)

    return output

def median_filtering(image, filter_w, filter_h):
    if filter_w % 2 == 0:
        filter_w += 1
    if filter_h % 2 == 0:
        filter_h += 1

    image_h, image_w = image.shape
    pad_w = filter_w // 2
    pad_h = filter_h // 2

    pad_image = numpy.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode = 'edge')
    output = numpy.zeros_like(image)

    for j in range(image_h):
        for i in range(image_w):
            area = pad_image[j:j + filter_h, i:i + filter_w]
            output[j,i] = numpy.median(area)

    return output

def hist_eq(image):
    image_h, image_w = image.shape
    total_pixels = image_h * image_w
    hist_arr = numpy.zeros(256)
    output = numpy.zeros_like(image)

    for i in range(image_h):
        for j in range(image_w):
            hist_arr[image[i,j]] += 1

    for i in range(1, 256):
        hist_arr[i] += hist_arr[i - 1]

    for i in range(image_h):
        for j in range(image_w):
            output[i,j] = numpy.uint8(hist_arr[image[i,j]] * 255 / total_pixels)

    return output

def rotate(image, theta):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h // 2
    center_w = image_w // 2

    rotated_w = round(abs(image_h * math.sin(theta)) + abs(image_w * math.cos(theta)))
    rotated_h = round(abs(image_h * math.cos(theta)) + abs(image_w * math.sin(theta)))

    rotated_image = numpy.zeros((rotated_h, rotated_w), dtype=numpy.uint8)

    new_center_h = rotated_h // 2
    new_center_w = rotated_w // 2

    for i in range(rotated_h):
        for j in range(rotated_w):
            original_x = int((j - new_center_w) * math.cos(-theta) - (i - new_center_h) * math.sin(-theta) + center_w)
            original_y = int((j - new_center_w) * math.sin(-theta) + (i - new_center_h) * math.cos(-theta) + center_h)

            if 0 <= original_x < image_w and 0 <= original_y < image_h:
                rotated_image[i,j] = image[original_y, original_x]

    return rotated_image

def edge_detection(image):
    sigma = 1.5
    filter_size = 5
    pad = 2
    gaus_filter = generate_gaussian(sigma, filter_size, filter_size)
    smooth_image = apply_filter(image, gaus_filter, pad, pad)

    enhance_image = hist_eq(smooth_image)

    threshold_value = 40
    threshold_image = numpy.where(enhance_image > threshold_value, 255, 0).astype(numpy.uint8)

    laplac_filter = numpy.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    edge_image = apply_filter(threshold_image, laplac_filter, 1, 0)

    return edge_image