#Import the necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np



#filter
def high_pass_filter(image, kernel_size=3):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高通滤波器
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(gray, -1, kernel)
    
    # 将滤波后的图像与原始图像相减得到高频部分
    high_pass_image = gray - filtered_image
    
    return high_pass_image

# garmma correction
def gamma_correction(image, gamma):
    # 归一化像素值到 [0, 1]
    normalized_image = image / 255.0

    # Gamma校正
    corrected_image = np.power(normalized_image, gamma)

    # 重新缩放像素值到 [0, 255]
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image

#his equal
def Histogram_equalization(path):
    image = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    img_eqhist=cv2.equalizeHist(image)
    
    compared = np.hstack((image,img_eqhist)) #stacking images side-by-side
    cv2.imwrite('compared.png',compared)
    
    hist=cv2.calcHist(image,[0],None,[256],[0,256])
    plt.subplot(121)
    plt.title("Image")
    plt.xlabel('bins')
    plt.ylabel("No of pixels")
    plt.plot(hist)
    
    hist2=cv2.calcHist(img_eqhist,[0],None,[256],[0,256])
    plt.subplot(122)
    plt.plot(hist2)
    plt.savefig("histogram_comparison.png")
    plt.show()



# create a CLAHE object (Arguments are optional).
def contrast_clahe(path):
    image = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    compared = np.hstack((image,cl1))
    cv2.imwrite('after_clahe.png',compared)

def smoothing(path):

    image = cv2.imread(path,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(image,(5,5))
    compared = np.hstack((image,blur)) #stacking images side-by-side
    cv2.imwrite('compared.png',compared)
    

from PIL import Image, ImageEnhance

image = Image.open('/home/ccy/cellcycle/data/split/train/brightfield/1919-900000002.tif')
factor = 3.0 #increase contrast
enhancer = ImageEnhance.Contrast(image)
im_output = enhancer.enhance(factor)


image = cv2.imread('/home/ccy/cellcycle/data/split/train/brightfield/1919-900000002.tif',cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl1 = clahe.apply(image)

plt.subplot(121)
plt.title("Image")
plt.imshow(cl1,cmap="gray")

plt.subplot(122)
plt.imshow(image,cmap="gray")
plt.savefig("PIL cv2.png")
plt.show()


