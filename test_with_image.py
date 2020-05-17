from LaneFinder import process_image
import matplotlib.pyplot as plt
import cv2


# select an image to work with
image = cv2.imread('test_images/test1.jpg')
#image = cv2.imread('test_images/straight_lines2.jpg')
#image = cv2.imread('test_images/debug_test2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_comb = process_image(image_rgb)

# plot the result
f, ax = plt.subplots(figsize=(12,9))
ax.imshow(img_comb)
f.tight_layout()
