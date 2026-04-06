import numpy as np
import cv2 as cv

video_file = 'chessboard.avi'

K = np.array([[956.49661865,   0.        , 961.37566344],
              [  0.        , 962.04059292, 536.96602399],
              [  0.        ,   0.        ,   1.        ]])

dist_coeff = np.array([-0.00342753, 0.01778027, -0.00133933, 0.00100904, -0.01872466])

video = cv.VideoCapture(video_file)
if not video.isOpened():
    exit()

map1, map2 = None, None

while True:
    valid, img = video.read()
    if not valid:
        break

    h, w = img.shape[:2]

    if map1 is None or map2 is None:
        map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (w, h), cv.CV_32FC1)

    rectified_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)

    cv.putText(img, "ORIGINAL", (50, 80), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
    cv.putText(rectified_img, "RECTIFIED", (50, 80), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)

    combined_view = np.hstack((img, rectified_img))
    display_img = cv.resize(combined_view, (w, h // 2))

    cv.imshow("Comparison", display_img)

    key = cv.waitKey(10)
    if key == ord(' '):
        cv.waitKey()
    elif key == 27:
        break

video.release()
cv.destroyAllWindows()