import numpy as np
import cv2
import pandas as pd
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from numpy.linalg import inv
from scipy import signal
from skimage import transform as tf
WINDOW_SIZE = 25

## Gaussian  2D function---------------------------------------------------------------------------------------
def GaussianPDF_1D(mu, sigma, length):
    # create an array
    half_len = length / 2

    if np.remainder(length, 2) == 0:                    # if divisible by 2
        ax = np.arange(-half_len, half_len, 1)          # negative and positive half around zero middle
    else:                                               # if not divisible by 2
        ax = np.arange(-half_len, half_len + 1, 1)      #  1 in the middle since odd length

    ax = ax.reshape([-1, ax.size])            
    # find numerator and denominator of the gaussian function
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )   

    return nominator / denominator

def GaussianPDF_2D(mu, sigma, row, col):
    # create row vector as 1D Gaussian pdf
    g_row = GaussianPDF_1D(mu, sigma, row)
    # create column vector as 1D Gaussian pdf
    g_col = GaussianPDF_1D(mu, sigma, col).transpose()

    return signal.convolve2d(g_row, g_col, 'full')      

def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

## interp2 utils---------------------------------------------------------------------------------------
def interp2(v, xq, yq):

    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'


    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor<0] = 0
    y_floor[y_floor<0] = 0
    x_ceil[x_ceil<0] = 0
    y_ceil[y_ceil<0] = 0

    x_floor[x_floor>=w-1] = w-1
    y_floor[y_floor>=h-1] = h-1
    x_ceil[x_ceil>=w-1] = w-1
    y_ceil[y_ceil>=h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h,q_w)
    return interp_val


#Compute harris corner interest points and return decriptor vectors.-----------------------------------------------
def getFeatures(img,bbox,use_shi=False):
    n_object = np.shape(bbox)[0]
    N = 0
    temp = np.empty((n_object,),dtype=np.ndarray)   # temporary storage of x,y coordinates
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:].astype(int))
        roi = img[ymin:ymin+boxh,xmin:xmin+boxw]
        # cv2.imshow('roi',roi)
        if use_shi:
            corner_response = corner_shi_tomasi(roi)
        else:
            corner_response = corner_harris(roi)
        coordinates = peak_local_max(corner_response,num_peaks=20,exclude_border=2)
        coordinates[:,1] += xmin
        coordinates[:,0] += ymin
        temp[i] = coordinates
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]
    x = np.full((N,n_object),-1)
    y = np.full((N,n_object),-1)
    for i in range(n_object):
        n_feature = temp[i].shape[0]
        x[0:n_feature,i] = temp[i][:,1]
        y[0:n_feature,i] = temp[i][:,0]
    return x,y


def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    n_object = bbox.shape[0]
    newbbox = np.zeros_like(bbox)
    Xs = newXs.copy()
    Ys = newYs.copy()
    for obj_idx in range(n_object):
        startXs_obj = startXs[:,[obj_idx]]
        startYs_obj = startYs[:,[obj_idx]]
        newXs_obj = newXs[:,[obj_idx]]
        newYs_obj = newYs[:,[obj_idx]]
        desired_points = np.hstack((startXs_obj,startYs_obj))
        actual_points = np.hstack((newXs_obj,newYs_obj))
        t = tf.SimilarityTransform()
        t.estimate(dst=actual_points, src=desired_points)
        mat = t.params

        # estimate the new bounding box with all the feature points
        # coords = np.vstack((bbox[obj_idx,:,:].T,np.array([1,1,1,1])))
        # new_coords = mat.dot(coords)
        # newbbox[obj_idx,:,:] = new_coords[0:2,:].T

        # estimate the new bounding box with only the inliners (Added by Yongyi Wang)
        THRES = 1
        projected = mat.dot(np.vstack((desired_points.T.astype(float),np.ones([1,np.shape(desired_points)[0]]))))
        distance = np.square(projected[0:2,:].T - actual_points).sum(axis = 1)
        actual_inliers = actual_points[distance < THRES]
        desired_inliers = desired_points[distance < THRES]
        if np.shape(desired_inliers)[0]<4:
            print('too few points')
            actual_inliers = actual_points
            desired_inliers = desired_points
        t.estimate(dst=actual_inliers, src=desired_inliers)
        mat = t.params
        coords = np.vstack((bbox[obj_idx,:,:].T,np.array([1,1,1,1])))
        new_coords = mat.dot(coords)
        newbbox[obj_idx,:,:] = new_coords[0:2,:].T
        Xs[distance >= THRES, obj_idx] = -1
        Ys[distance >= THRES, obj_idx] = -1

    return Xs, Ys, newbbox

def estimateAllTranslation(startXs,startYs,img1,img2):
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I,(5,5),0.2)
    Iy, Ix = np.gradient(I.astype(float))

    startXs_flat = startXs.flatten()
    startYs_flat = startYs.flatten()
    newXs = np.full(startXs_flat.shape,-1,dtype=float)
    newYs = np.full(startYs_flat.shape,-1,dtype=float)
    for i in range(np.size(startXs)):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2)
    newXs = np.reshape(newXs, startXs.shape)
    newYs = np.reshape(newYs, startYs.shape)
    return newXs, newYs


def estimateFeatureTranslation(startX,startY,Ix,Iy,img1,img2):
    X=startX
    Y=startY
    mesh_x,mesh_y=np.meshgrid(np.arange(WINDOW_SIZE),np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mesh_x_flat_fix =mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix =mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix,mesh_y_flat_fix))
    I1_value = interp2(img1_gray, coor_fix[[0],:], coor_fix[[1],:])
    Ix_value = interp2(Ix, coor_fix[[0],:], coor_fix[[1],:])
    Iy_value = interp2(Iy, coor_fix[[0],:], coor_fix[[1],:])
    I=np.vstack((Ix_value,Iy_value))
    A=I.dot(I.T)
   

    for _ in range(15):
        mesh_x_flat=mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat=mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor=np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip=(I2_value-I1_value).reshape((-1,1))
        b=-I.dot(Ip)
        solution=inv(A).dot(b)
        # solution = np.linalg.solve(A, b)
        X += solution[0,0]
        Y += solution[1,0]
    
    return X, Y
