'''
Corner detection and perspective correction
Jan Ondras
22/10/2017
'''

import numpy as np
import cv2
import glob

images = glob.glob('./img/*.jpg')
new_width = new_height = 250

# Load previously obtained camera calibration parameters
#calibration_data = np.load('./camera_config.npz')

for img_name in images:
    img_org = cv2.imread(img_name)
    img_org = cv2.resize(img_org, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    # Remove distortion
    #img_org = cv2.undistort(img_org, calibration_data['camera_matrix'], calibration_data['dist_coefs'])
    img_org_show = img_org.copy()

    gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (13, 13), 0)
    edges = cv2.Canny(gray, 75, 200)
    cv2.imshow("Edges", edges)
    cv2.imwrite('example_edged.png', edges)

    # Heuristic: assume that the largest contour in the image with exactly four points is the outer border 
    # of the stamp. Find contours in the edged image, sort them by area enclosed by the contour keeping only
    # few largest ones. "CHAIN_APPROX_SIMPLE" removes all redundant contour points => contour is compressed
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
     
    # Iterate over contours
    for c in contours:
        # Approximate the contour (tuning epsilon)
        epsilon = 0.05 * cv2.arcLength(c, True)
        c_approx = cv2.approxPolyDP(c, epsilon, True)
        #print len(c_approx)
        #cv2.drawContours(img_org_show, [c], -1, (0, 255, 0), 1)

        # If approximating contour has 4 points, then assume that outer border of the stamp was detected
        if len(c_approx) == 4:
            # Show non-approximating contour (green)
            cv2.drawContours(img_org_show, [c], -1, (0, 255, 0), 1)
            # Show approximating contour (red)
            cv2.drawContours(img_org_show, [c_approx], -1, (0, 0, 255), 2)
            stamp_contour = c_approx.reshape(4, 2)
            break

    # To keep consistent order of corners: top-left, top-right, bottom-right, bottom-left
    corners_detected = np.zeros((4, 2), dtype = "float32")
    sum_xy = stamp_contour.sum(axis=1)
    corners_detected[0] = stamp_contour[np.argmin(sum_xy)]  # top-left
    corners_detected[2] = stamp_contour[np.argmax(sum_xy)]  # bottom-right
    sub_xy = np.diff(stamp_contour, axis = 1)
    corners_detected[1] = stamp_contour[np.argmin(sub_xy)]  # top-right
    corners_detected[3] = stamp_contour[np.argmax(sub_xy)]  # bottom-left

    # Desired image plane points
    corners_transformed = np.array([
        [0, 0], 
        [new_width-1, 0], 
        [new_width-1, new_height-1], 
        [0,  new_height-1]], dtype='float32')

    # Obtain and apply perspective transform matrix Pt
    Pt = cv2.getPerspectiveTransform(corners_detected, corners_transformed)
    img_corr = cv2.warpPerspective(img_org, Pt, (new_width, new_height))

    # Show original and perspective corrected images
    cv2.imwrite('example_contours.png', img_org_show)
    cv2.imwrite('example_corrected.png', img_corr)
    cv2.imshow("Original image", img_org_show)
    cv2.imshow("Perspective corrected image", img_corr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




'''
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

https://docs.opencv.org/3.3.0/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html


https://docs.opencv.org/3.3.0/da/d22/tutorial_py_canny.html
'''




'''
import numpy as np
import cv2
import glob

images = glob.glob('./img/*.jpg')
new_width = new_height = 250

for img_name in images:
    img_org = cv2.imread(img_name)
    img_org = cv2.resize(img_org, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    #h, w = img_org.shape[0:2]
    print img_org.shape[0:2]

    gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 75, 200)
    cv2.imshow("Original image", img_org)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)

    #The heuristic goes something like this: we ll assume that the largest contour in the image with exactly four points is our piece of paper to be scanned.
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # removes all redundant points and compresses the contour, thereby saving memory
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    #print len(contours)
     
    # loop over the contours
    for c in contours:
        # approximate the contour
        epsilon = 0.05 * cv2.arcLength(c, True)
        c_approx = cv2.approxPolyDP(c, epsilon, True)

        cv2.drawContours(img_org, [c], -1, (0, 255, 0), 1)
        cv2.drawContours(img_org, [c_approx], -1, (0, 0, 255), 1)
        cv2.imshow("Contour", img_org)
        cv2.waitKey(0)
        print len(c), len(c_approx)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(c_approx) == 4:
            stamp_contour = c_approx
            print "found first with 4"
            break

    # show the contour (outline) of the piece of paper
    cv2.drawContours(img_org, [stamp_contour], -1, (255, 255, 0), 2)
    cv2.imshow("Contour", img_org)
    cv2.waitKey(0)
    #break
    stamp_contour = stamp_contour.reshape(4, 2)
    print stamp_contour
    # Consistent order: top-left, top-right, bottom-right, bottom-left
    corners_detected = np.zeros((4, 2), dtype = "float32")
    sum_xy = stamp_contour.sum(axis=1)
    corners_detected[0] = stamp_contour[np.argmin(sum_xy)]  # top-left
    corners_detected[2] = stamp_contour[np.argmax(sum_xy)]  # bottom-right
    sub_xy = np.diff(stamp_contour, axis = 1)
    corners_detected[1] = stamp_contour[np.argmin(sub_xy)]  # top-right
    corners_detected[3] = stamp_contour[np.argmax(sub_xy)]  # bottom-left

    # Desired image plane points
    corners_transformed = np.array([
        [0, 0], 
        [new_width-1, 0], 
        [new_width-1, new_height-1], 
        [0,  new_height-1]], dtype='float32')

    Pt = cv2.getPerspectiveTransform(corners_detected, corners_transformed)
    img_corr = cv2.warpPerspective(img_org, Pt, (new_width, new_height))

    #break

    # Show original and perspective corrected images
    cv2.imshow("Original image", img_org)
    cv2.imshow("Perspective corrected image", img_corr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''