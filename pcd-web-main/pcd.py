import cv2;
import numpy as np

# read image with color
def readImage(path):
    img = cv2.imread(path)
    return img

# show image
def showImage(img, windowName):
    cv2.imshow(windowName, img)

# extract RGB from BGR Image
def extractRGB(img):
    rgb_channel = [img[:, :, 2], img[:, :, 1], img[:, :, 0]]

    return rgb_channel

# Weighted Average Method To Convert Color Image To Grayscale
def weightedAverageGrayscale(img):
    [R, G, B] = extractRGB(img)
    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)

    grayscale = img.copy()

    grayscale[:, :, 0] = grayscale[:, :, 1]  = grayscale[:, :, 2] = Y

    return grayscale

# Luminousity Grayscale Method
def luminosityGrayscale(img):
    [R, G, B] = extractRGB(img)
    Z = 0.2126 * R + 0.7152 * G + 0.0722 * B

    grayscale = img.copy()

    grayscale[:, :, 0] = grayscale[:, :, 1]  = grayscale[:, :, 2] = Z

    return grayscale

# Average Grayscale Method
def averageGrayscale(img):
    [row, col] = img.shape[0:2]

    for i in range(row):
        for j in range(col):
            img[i, j] = sum(img[i, j] * 0.33)
    
    return img

def convertToGrayscale(img):
    [R, G, B] = extractRGB(img)
    grayscale = (R + G + B) / 3

    img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = grayscale

    return img

def detectCircleByColor(img, circle_color):
    [row, col] = img.shape[0:2]

    for i in range(row):
        for j in range(col):
            if(circle_color in img[i, j]):
                img[i, j] = [0, 0, 255]
    return img

# Brightness
# def brightnessAddSub(image, value):
#     image = np.asarray(image).astype('int16')
#     image = image+value
#     image = np.clip(image, 0, 255)
#     new_image = image.astype('uint8') 
#     return new_image

# def brightness_multiplication(image, value):
#     image = np.asarray(image).astype('uint16')
#     image = image*value
#     image = np.clip(image, 0, 255)
#     new_image = image.astype('uint8') 
#     return new_image

# def brightness_divide(image, value):
#     image = np.asarray(image).astype('uint16')
#     image = image/value
#     image = np.clip(image, 0, 255)
#     new_image = image.astype('uint8') 
#     return new_image

# def brightnessOpenCV(image, value):
#     print(image[:,:,0])
#     if value >= 0:
#         image[:, :, 0] = cv2.add(image[:, :, 0], value)
#         image[:, :, 1] = cv2.add(image[:, :, 1], value)
#         image[:, :, 2] = cv2.add(image[:, :, 2], value)
#     else:
#         image[:, :, 0] = cv2.subtract(image[:, :, 0], -value)
#         image[:, :, 1] = cv2.subtract(image[:, :, 1], -value)
#         image[:, :, 2] = cv2.subtract(image[:, :, 2], -value)

#     new_image = np.clip(image, 0, 255)

#     return new_image

# def brightness_multiplicationcv(image, value):
#     image[:, :, 0] = cv2.multiply(image[:, :, 0], value)
#     image[:, :, 1] = cv2.multiply(image[:, :, 1], value)
#     image[:, :, 2] = cv2.multiply(image[:, :, 2], value)
#     new_image= np.clip(image, 0, 255)
#     return new_image

# def brightness_dividecv(image, value):
#     print(image)
#     image[:, :, 0] = cv2.divide(image[:, :, 0], value)
#     image[:, :, 1] = cv2.divide(image[:, :, 1], value)
#     image[:, :, 2] = cv2.divide(image[:, :, 2], value)
#     new_image= np.clip(image, 0, 255)
#     return new_image

# def make_dimension_equal(image1, image2, x_offset=0, y_offset=0):
#     height1, width1 = image1.shape[:2]
#     height2, width2 = image2.shape[:2]

#     max_height = max(height1, height2)
#     max_width = max(width1, width2)

#     blank_image = np.zeros((max_height,max_width,3), np.uint8)
#     blank_image[:,:] = (0,0,0)

#     new_image_1 = blank_image.copy()                   
#     new_image_2 = blank_image.copy()                    

#     new_image_1[y_offset:y_offset+height1, x_offset:x_offset+width1] = image1.copy()
#     new_image_2[y_offset:y_offset+height2, x_offset:x_offset+width2] = image2.copy()

#     return [new_image_1, new_image_2]


# def bitwise_and(image1, image2):
#     [new_image_1, new_image_2] = make_dimension_equal(image1, image2)
#     bit_and = cv2.bitwise_and(new_image_1, new_image_2)
#     return bit_and

# def bitwise_or(image1, image2):
#     [new_image_1, new_image_2] = make_dimension_equal(image1, image2)
#     bit_or = cv2.bitwise_or(new_image_1, new_image_2)
#     return bit_or

# def bitwise_not(image):
#     bit_not = cv2.bitwise_not(image)
#     return bit_not

# def bitwise_xor(image1, image2):
#     [new_image_1, new_image_2] = make_dimension_equal(image1, image2)
#     bit_xor = cv2.bitwise_xor(new_image_1, new_image_2)
#     return bit_xor