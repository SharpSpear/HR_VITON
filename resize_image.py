import cv2
import glob

# img = cv2.imread("/root/HR-VITON/data/test/cloth/cloth.jpg")
# img = cv2.resize(img, dsize=(768, 1024))
# print(img.shape)
# img = cv2.imwrite("/root/HR-VITON/data/test/cloth/cloth_.jpg", img)

dst_h = 1024
dst_w = 768

list_imgs = glob.glob("./data/test/image/*")
for path in list_imgs:
    img = cv2.imread(path)
    h, w, _ = img.shape[:]
    ratio_h = dst_h / h 
    ratio_w = dst_w / w 
    ratio = min(ratio_h, ratio_w)
    img = cv2.resize(img, fx = ratio, fy = ratio, dsize = None)
    h, w, _ = img.shape[:]
    img = cv2.copyMakeBorder(img, 0, dst_h - h, 0, dst_w - w, cv2.BORDER_REPLICATE)
    print(img.shape)
    # print(path)
    cv2.imwrite(path, img)
    cv2.imwrite(path.replace("image", "cloth"), img)
    cv2.imwrite(path.replace("image", "cloth-mask"), img)

    # print(img.shape)

# list_imgs = glob.glob("./data/test/cloth/*")
# for path in list_imgs:
#     img = cv2.imread(path)
#     print(path)
#     img = cv2.resize(img, dsize=(768, 1024))
#     cv2.imwrite(path, img)
#     print(img.shape)