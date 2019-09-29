import cv2
import os



img = cv2.imread('Atest', cv2.IMREAD_GRAYSCALE)
print img
filenames = loadImages(pathfile='Card_Imgs')
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))



#diff_img = cv2.absdiff(img, images[0])
#print int(np.sum(diff_img) / 255)  # 1562

cv2.imshow("Original", img)
cv2.waitKey()
cv2.destroyAllWindows()

def loadImages(pathfile):
    return [os.path.join(pathfile, f) for f in os.listdir(pathfile) if f.endswith('jpg')]