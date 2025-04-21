import numpy as np
import cv2
import random

def conv2d(img, kernel):
    h, w = img.shape
    m, n = kernel.shape
    pad_img = np.pad(img, (m//2, n//2), 'edge').astype(np.float64)
    result = np.zeros_like(img)
    for i in range(m):
        for j in range(n):
            result = result + pad_img[i:h+i, j:w+j]*kernel[i, j]

    return result

def solve_homography(u, v):
    N = u.shape[0]
    A = np.zeros((2*N, 9))
    for i in range(N):
        ux = u[i, 0]
        uy = u[i, 1]
        vx = v[i, 0]
        vy = v[i, 1]
        A[2*i, :] = np.array([ux, uy, 1, 0, 0, 0, -ux*vx, -uy*vx, -vx])
        A[2*i+1, :] = np.array([0, 0, 0, ux, uy, 1, -ux*vy, -uy*vy, -vy])

    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape((3, 3))

    return H

def rectangling(img, corners, n, oh, ow):

    h, w, c = img.shape
    th = oh*3
    tw = ow*(n+1)
    tcorners = [(0, 0), (th-1, 0), (0, tw-1), (th-1, tw-1)]
    u = np.ones((4, 3))
    v = np.ones((4, 3))
    for i in range(4):
        u[i, 0] = corners[i][0]
        u[i, 1] = corners[i][1]
        v[i, 0] = tcorners[i][0]
        v[i, 1] = tcorners[i][1]

    H = solve_homography(u, v)
    result = np.zeros((th, tw, 3))
    yy, xx = np.meshgrid(np.arange(0, tw), np.arange(0, th))
    N = xx.shape[0] * xx.shape[1]
    U = np.array([xx.flatten(), yy.flatten(), np.ones(N)])
    origin = np.matmul(np.linalg.inv(H), U)
    originX = (origin[0] / origin[2]).reshape(xx.shape)
    originY = (origin[1] / origin[2]).reshape(yy.shape)

    baseX = np.floor(originX).astype(np.int32)
    baseY = np.floor(originY).astype(np.int32)
    rx = originX-baseX
    ry = originY-baseY
    rx = np.repeat(np.expand_dims(rx, axis = -1), 3, axis = -1)
    ry = np.repeat(np.expand_dims(ry, axis = -1), 3, axis = -1)

    xmask = np.logical_and(baseX >= 0, baseX < h-1)
    ymask = np.logical_and(baseY >= 0, baseY < w-1)
    mask = np.logical_and(xmask, ymask)

    patch = (1-rx)*(1-ry)*img[(baseX*mask, baseY*mask)] + \
            (1-rx)*ry*img[(baseX*mask, (baseY+1)*mask)] + \
            (1-ry)*rx*img[((baseX+1)*mask, baseY*mask)] + \
            rx*ry*img[((baseX+1)*mask, (baseY+1)*mask)]

    result[0:th, 0:tw][mask] = patch[mask]

    low_h = 0
    high_h = 0
    for i in range(th):
        if np.sum(result[i, :, :] == [0, 0, 0]) == 0:
            low_h = i
            break
    for i in range(th):
        if np.sum(result[th-1-i, :, :] == [0, 0, 0]) == 0:
            high_h = th-1-i
            break
    cropped_result = result[low_h:high_h+1, :]

    return result, cropped_result

# feature detection
def HarrisCorner(img):
    h, w, c = img.shape
    Gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Hx = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    Hy = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    Ix = conv2d(Gimg, Hx)
    Iy = conv2d(Gimg, Hy)

    m, theta = calc_orientation(Ix, Iy)

    Ix2 = Ix ** 2
    Iy2 = Iy **2
    Ixy = Ix * Iy

    size = 21
    sigma = 2
    Sx2 = cv2.GaussianBlur(Ix2, (size, size), sigma)
    Sy2 = cv2.GaussianBlur(Iy2, (size, size), sigma)
    Sxy = cv2.GaussianBlur(Ixy, (size, size), sigma)

    k = 0.05
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - k*traceM

    thres = np.max(R)*0.02
    candidates = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            if R[i, j] <= thres:
                continue
            if np.max(R[i-1:i+2, j-1:j+2]) == R[i, j]:
                candidates.append((i, j, R[i, j]))
    
    def sortkey(e):
        return e[2]

    candidates.sort(reverse=True, key=sortkey)
    candidates = candidates[:200]

    feature_points = []
    descriptors = []
    for can in candidates:
        point = (can[0], can[1])
        feature_points.append(point)
        descriptors.append(gen_descriptor(can[0], can[1], m, theta))

    return np.array(feature_points), np.array(descriptors)

def simple_descriptor(fpx, fpy, img):
    patch = img[fpx-1:fpx+2, fpy-1:fpy+2, :]
    descriptor = patch.flatten()

    return descriptor

def calc_orientation(Ix, Iy):

    m = np.sqrt(Ix ** 2 + Iy ** 2)

    theta = np.arctan2(Iy, Ix) * 180/np.pi
    theta = (theta+360)%360

    return m, theta

def gen_descriptor(fpx, fpy, m, theta):
    m = np.pad(m, ((7, 8), (7, 8)), 'edge')
    theta = np.pad(theta, ((7, 8), (7, 8)), 'edge')
    patchM = m[fpx+3:fpx+12, fpy+3:fpy+12]
    patchM = cv2.GaussianBlur(patchM, (9, 9), 1.5*3)
    patchT = theta[fpx+3:fpx+12, fpy+3:fpy+12]

    bins = 8
    angles = np.zeros((bins))
    for i in range(bins):
        low = 360*i/bins
        high = 360*(i+1)/bins
        mask = np.logical_and(patchT >= low, patchT < high)
        angles[i] = np.sum(patchM[mask])
    main_theta = 360*(np.argmax(angles)+0.5)/bins

    descriptors = []
    patchT = theta[fpx:fpx+16, fpy:fpy+16] - main_theta
    patchM = m[fpx:fpx+16, fpy:fpy+16]
    for i in range(4):
        for j in range(4):
            patcht = patchT[i*4:(i+1)*4, j*4:(j+1)*4]
            patchm = patchM[i*4:(i+1)*4, j*4:(j+1)*4]
            for k in range(bins):
                low = 360*k/bins
                high = 360*(k+1)/bins
                mask = np.logical_and(patcht >= low, patcht < high)
                descriptors.append(np.sum(patchm[mask]))
    descriptors = np.array(descriptors)
    descriptors = descriptors/np.max(descriptors)
    descriptors = np.clip(descriptors, 0, 0.2)
    descriptors = descriptors/np.max(descriptors)

    return descriptors

# feature matching
def MatchFeature(fp1, des1, fp2, des2, width):
    n1, _ = fp1.shape
    n2, _ = fp2.shape
    matches = []

    for i in range(n1):
        if fp1[i][1] < width/2:
            continue
        dist0 = np.linalg.norm(des1[i]-des2[0])
        dist1 = np.linalg.norm(des1[i]-des2[1])
        best = 0 if dist0 < dist1 else 1
        best_dist1 = dist0 if dist0 < dist1 else dist1
        best_dist2 = dist1 if dist0 < dist1 else dist0
        for j in range(2, n2):
            dist = np.linalg.norm(des1[i]-des2[j])
            if dist < best_dist1:
                best_dist2 = best_dist1
                best = j
                best_dist1 = dist
            elif dist < best_dist2:
                best_dist2 = dist
        score = best_dist2/best_dist1
        matches.append([fp1[i], fp2[best], score])
    
    def sortkey(e):
        return e[2]

    matches.sort(reverse=True, key=sortkey)

    return matches

# image matching
def Ransac(matches, seed):
    n = len(matches)
    kp1 = np.ones((n, 3))
    kp2 = np.ones((n, 3))
    for i in range(n):
        kp1[i, 0], kp1[i, 1] = matches[i][0]
        kp2[i, 0], kp2[i, 1] = matches[i][1]
    
    N = 1000
    best_inliers = 0
    best_H = None
    random.seed(seed)
    good = min(max(n//4, 12), n)

    for _ in range(N):
        samples = random.sample(range(good), 4)
        u = kp2[samples]
        v = kp1[samples]
        H = solve_homography(u, v)
        U = kp2.T
        V = kp1.T
        HU = np.matmul(H, U)

        if np.sum(HU[-1] == 0):
            continue

        dist = np.linalg.norm((HU / HU[-1])-V, axis = 0, ord = 1)
        mask = dist[:good] < 4
        inliers = np.sum(mask)

        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H

    return best_H

# blending
def BlendR(img1, img2, H, offset_h, offset_w):

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    H_inv = np.linalg.inv(H)

    corners = np.array([[0, 0, h2-1, h2-1], [0, w2-1, 0, w2-1], [1, 1, 1, 1]])
    corners = np.matmul(H, corners)
    corners = corners / corners[-1]
    minH = np.ceil(np.min(corners[0])).astype(np.int32)+offset_h
    maxH = np.floor(np.max(corners[0])).astype(np.int32)+offset_h
    minW = np.ceil(np.min(corners[1])).astype(np.int32)+offset_w
    maxW = np.floor(np.max(corners[1])).astype(np.int32)+offset_w

    sh = max(h1, maxH)
    sw = max(w1, maxW)
    offset_now = 0
    if minH < 0:
        sh -= minH
        offset_h -= minH
        offset_now = -minH

    result = np.zeros((sh, sw, 3))
    result[offset_now:h1+offset_now, :w1] = img1

    yy, xx = np.meshgrid(np.arange(minW, maxW), np.arange(minH+offset_now, maxH+offset_now))
    N = xx.shape[0] * xx.shape[1]
    xx = xx - offset_h
    yy = yy - offset_w
    U = np.array([xx.flatten(), yy.flatten(), np.ones(N)])

    origin = np.matmul(H_inv, U)
    originX = (origin[0] / origin[2]).reshape(xx.shape)
    originY = (origin[1] / origin[2]).reshape(yy.shape)

    baseX = np.floor(originX).astype(np.int32)
    baseY = np.floor(originY).astype(np.int32)
    rx = originX-baseX
    ry = originY-baseY
    rx = np.repeat(np.expand_dims(rx, axis = -1), 3, axis = -1)
    ry = np.repeat(np.expand_dims(ry, axis = -1), 3, axis = -1)

    xmask = np.logical_and(baseX >= 0, baseX < h2-1)
    ymask = np.logical_and(baseY >= 0, baseY < w2-1)
    mask = np.logical_and(xmask, ymask)

    patch = (1-rx)*(1-ry)*img2[(baseX*mask, baseY*mask)] + \
            (1-rx)*ry*img2[(baseX*mask, (baseY+1)*mask)] + \
            (1-ry)*rx*img2[((baseX+1)*mask, baseY*mask)] + \
            rx*ry*img2[((baseX+1)*mask, (baseY+1)*mask)]

    ratio1 = w1 - yy
    ratio1[ratio1 < 0] = 0
    zero = np.logical_and(result[minH+offset_now:maxH+offset_now, minW:maxW, 0] == 0, result[minH+offset_now:maxH+offset_now, minW:maxW, 1] == 0)
    zero = np.logical_and(zero, result[minH+offset_now:maxH+offset_now, minW:maxW, 2] == 0)
    ratio1[zero] = 0
    ratio2 = originY
    ratiop = ratio2/(ratio1+ratio2)
    ratior = ratio1/(ratio1+ratio2)
    ratiop = np.repeat(np.expand_dims(ratiop, axis = -1), 3, axis = -1)
    ratior = np.repeat(np.expand_dims(ratior, axis = -1), 3, axis = -1)

    result[minH+offset_now:maxH+offset_now, minW:maxW][mask] = patch[mask]*ratiop[mask] + result[minH+offset_now:maxH+offset_now, minW:maxW][mask]*ratior[mask]

    return result, offset_h

def BlendL(img1, img2, H, offset_h, offset_w):
    
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    H_inv = np.linalg.inv(H)

    corners = np.array([[0, 0, h2-1, h2-1], [0, w2-1, 0, w2-1], [1, 1, 1, 1]])
    corners = np.matmul(H_inv, corners)
    corners = corners / corners[-1]
    minH = np.ceil(np.min(corners[0])).astype(np.int32)+offset_h
    maxH = np.floor(np.max(corners[0])).astype(np.int32)+offset_h
    minW = np.ceil(np.min(corners[1])).astype(np.int32)+offset_w
    maxW = np.floor(np.max(corners[1])).astype(np.int32)+offset_w

    sh = max(h1, maxH)
    sw = w1-minW

    offset_w -= minW
    offset_now = 0
    if minH < 0:
        sh -= minH
        offset_h -= minH
        offset_now = -minH

    result = np.zeros((sh, sw, 3))
    result[offset_now:h1+offset_now, -minW:w1-minW] = img1

    yy, xx = np.meshgrid(np.arange(0, maxW-minW), np.arange(minH+offset_now, maxH+offset_now))
    N = xx.shape[0] * xx.shape[1]
    xx = xx - offset_h
    yy = yy - offset_w
    U = np.array([xx.flatten(), yy.flatten(), np.ones(N)])

    origin = np.matmul(H, U)
    originX = (origin[0] / origin[2]).reshape(xx.shape)
    originY = (origin[1] / origin[2]).reshape(yy.shape)

    baseX = np.floor(originX).astype(np.int32)
    baseY = np.floor(originY).astype(np.int32)
    rx = originX-baseX
    ry = originY-baseY
    rx = np.repeat(np.expand_dims(rx, axis = -1), 3, axis = -1)
    ry = np.repeat(np.expand_dims(ry, axis = -1), 3, axis = -1)

    xmask = np.logical_and(baseX >= 0, baseX < h2-1)
    ymask = np.logical_and(baseY >= 0, baseY < w2-1)
    mask = np.logical_and(xmask, ymask)

    patch = (1-rx)*(1-ry)*img2[(baseX*mask, baseY*mask)] + \
            (1-rx)*ry*img2[(baseX*mask, (baseY+1)*mask)] + \
            (1-ry)*rx*img2[((baseX+1)*mask, baseY*mask)] + \
            rx*ry*img2[((baseX+1)*mask, (baseY+1)*mask)]

    ratio1 = yy
    ratio1[ratio1 < 0] = 0
    zero = np.logical_and(result[minH+offset_now:maxH+offset_now, 0:maxW-minW, 0] == 0, result[minH+offset_now:maxH+offset_now, 0:maxW-minW, 1] == 0)
    zero = np.logical_and(zero, result[minH+offset_now:maxH+offset_now, 0:maxW+-minW, 2] == 0)
    ratio1[zero] = 0
    ratio2 = w2 - originY
    ratiop = ratio2/(ratio1+ratio2)
    ratior = ratio1/(ratio1+ratio2)
    ratiop = np.repeat(np.expand_dims(ratiop, axis = -1), 3, axis = -1)
    ratior = np.repeat(np.expand_dims(ratior, axis = -1), 3, axis = -1)

    result[minH+offset_now:maxH+offset_now, 0:maxW-minW][mask] = patch[mask]*ratiop[mask] + result[minH+offset_now:maxH+offset_now, 0:maxW-minW][mask]*ratior[mask]

    return result, offset_h, offset_w