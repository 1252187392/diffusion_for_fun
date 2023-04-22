import matplotlib.pyplot as plt
import cv2

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

def resize_with_min_size(im, target_size):
    '''
    固定resize，短边resize到min_size,短边对应放缩
    target_size (w,h)
    '''
    h, w, _ = im.shape
    sc = [im.shape[1] / target_size[0], im.shape[0] / target_size[1]]
    if sc[0] >= sc[1]:
        h = target_size[1]
        w = int(w / sc[1])
    else:
        w = target_size[0]
        h = int(h / sc[0])
    #new_w = int(new_w//32*32)
    #new_h = int(new_h//32*32)
    im = cv2.resize(im,(w, h))
    return im