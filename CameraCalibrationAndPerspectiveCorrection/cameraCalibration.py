'''
Camera Calibration & Distortion Removal
Jan Ondras
22/10/2017
'''

import numpy as np
import cv2
import glob

images = glob.glob('./imgCalib/*.jpg')
image_to_undistort = './imgCalib/IMG_20171028_174106.jpg'

square_size = 1.0
pattern_size = (9, 6)

# Prepare pattern points in model world coord. (z=0): (0,0,0), (1,0,0), ... (1,2,0) ...
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

# Arrays to store object points and image points from all images
obj_points = [] # 3D points in real world space
img_points = [] # 2D points in image plane

for image_name in images:
    img = cv2.imread(image_name)
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    # If found, add object points, image points (after refining them)
    if found == True:
        obj_points.append(pattern_points)
        
        # Refine corners (search window = 23x23)
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners)
        
        # Draw the corners
        cv2.drawChessboardCorners(img, pattern_size, corners, found)
        cv2.imshow('Chessboard corners: ' + image_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print 'Chessboard not found! ', image_name
    continue

# Calibrate camera to get camera matrix and distortion coefficients
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, 
                                                                   gray.shape[::-1], None, None)
print "RMS:", rms
print "Camera matrix:\n", camera_matrix
print "Rotation vectors:\n", rvecs
print "Translation vectors:\n", tvecs
print "Distortion coefficients:\n", dist_coefs.ravel()

# Save the camera matrix and distortion coefficients for future use
np.savez('./camera_config.npz', camera_matrix=camera_matrix, dist_coefs=dist_coefs)

# Distortion removal
img2 = cv2.imread(image_to_undistort)
img2 = cv2.resize(img2, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
res = cv2.undistort(img2, camera_matrix, dist_coefs)
cv2.imshow('Distorted', img2)
cv2.imshow('Undistorted', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
