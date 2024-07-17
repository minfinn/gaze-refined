import cv2
import matplotlib.pyplot as plt

# # 创建一个大小为6*4英寸，分辨率为80dpi的画布
# fig = plt.figure(figsize=(2560, 1600), dpi=10)
#
#
#
#
# # 保存图像，分辨率为120dpi
# plt.savefig('test.png', dpi=1)
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 255, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness = 1)
        cv2.imshow("image", img)

img = cv2.imread("test.png")
cv2.namedWindow("image")
cv2.imshow("image", img)
cv2.resizeWindow("image", 2560, 1600)
cv2.setMouseCallback("image", mouse)

cv2.waitKey(0)
cv2.destroyAllWindows()