import numpy as np
import cv2
from skimage.feature import corner_harris, corner_subpix
from skimage.draw import disk
from functions import img_to_8bit


def block_harris_corner(img, n_rows=10, n_cols=10, top_k=50, method="k", k=0.05, eps=1e-6, sigma=1, min_dist=10):
    harris_corners = []
    row_step = img.shape[0] // n_rows
    col_step = img.shape[1] // n_cols
    for i in range(0, img.shape[0], row_step):
        for j in range(0, img.shape[1], col_step):
            row_end = i + row_step if i + row_step < img.shape[0] else img.shape[0]
            col_end = j + col_step if j + col_step < img.shape[1] else img.shape[1]

            img_block = img[i:row_end, j:col_end]
            harris_image = corner_harris(img_block, method=method, k=k, eps=eps, sigma=sigma)

            corners = np.empty(shape=(top_k, 2), dtype=np.int32)
            for max_idx in range(top_k):
                idx = np.argmax(harris_image)
                idx_tuple = np.unravel_index(idx, shape=harris_image.shape)
                corners[max_idx] = np.array(idx_tuple)
                rr, cc = disk(corners[max_idx], radius=min_dist, shape=harris_image.shape)
                harris_image[rr, cc] = np.NINF

            corners = corners + np.array([i, j])
            harris_corners.append(corners)

    harris_corners = np.concatenate(harris_corners, axis=0)

    return harris_corners


def detect_block_harris_and_match(f_gray, m_gray, f_img, m_img, n_rows=10, n_cols=10, top_k=50, method="k", k=0.05,
                                  eps=1e-6,
                                  sigma=1,
                                  min_dist=10,
                                  subpix=False, window_size=9, alpha=0.99,
                                  sift_kp_size=9, match_ratio=0.7):
    """
    Compute Harris corners, then compute the corners' SIFT descriptors,
        perform matching to generate the initial CPs.
    """
    f_harris = block_harris_corner(f_gray, n_rows, n_cols, top_k, method, k, eps, sigma, min_dist)
    m_harris = block_harris_corner(m_gray, n_rows, n_cols, top_k, method, k, eps, sigma, min_dist)

    if subpix:
        f_harris = corner_subpix(f_gray, f_harris, window_size, alpha)
        m_harris = corner_subpix(m_gray, m_harris, window_size, alpha)

    f_8bit = img_to_8bit(f_img)
    m_8bit = img_to_8bit(m_img)

    f_harris_kp = [cv2.KeyPoint(x, y, sift_kp_size) for [y, x] in f_harris.tolist()]
    m_harris_kp = [cv2.KeyPoint(x, y, sift_kp_size) for [y, x] in m_harris.tolist()]

    sift = cv2.xfeatures2d.SIFT_create()
    f_sift_kp, f_des = sift.compute(f_8bit, f_harris_kp)
    m_sift_kp, m_des = sift.compute(m_8bit, m_harris_kp)

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(f_des, m_des, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        if m1.distance < match_ratio * m2.distance:
            good_matches.append([m1])

    f_pts = []
    m_pts = []
    for good_match in good_matches:
        idx1 = good_match[0].queryIdx
        idx2 = good_match[0].trainIdx
        f_pts.append([f_sift_kp[idx1].pt[1], f_sift_kp[idx1].pt[0]])
        m_pts.append([m_sift_kp[idx2].pt[1], m_sift_kp[idx2].pt[0]])

    f_pts = np.array(f_pts)
    m_pts = np.array(m_pts)

    print(f"Detected {f_harris.shape[0]} harris corners, matched {f_pts.shape[0]} using the SIFT descriptor.")

    return f_pts, m_pts


def detect_block_harris_and_match1(f_gray, m_gray, f_img, m_img, n_rows=10, n_cols=10, top_k=50, method="k", k=0.05,
                                   eps=1e-6,
                                   sigma=1,
                                   min_dist=10,
                                   sift_kp_size=9, match_ratio=0.7):
    f_8bit = img_to_8bit(f_img)
    m_8bit = img_to_8bit(m_img)
    f_pts = []
    m_pts = []
    row_step = f_gray.shape[0] // n_rows
    col_step = f_gray.shape[1] // n_cols
    for i in range(0, f_gray.shape[0], row_step):
        for j in range(0, f_gray.shape[1], col_step):
            row_end = i + row_step if i + row_step < f_gray.shape[0] else f_gray.shape[0]
            col_end = j + col_step if j + col_step < f_gray.shape[1] else f_gray.shape[1]

            f_block = f_gray[i:row_end, j:col_end]
            f_harris = corner_harris(f_block, method=method, k=k, eps=eps, sigma=sigma)

            m_block = m_gray[i:row_end, j:col_end]
            m_harris = corner_harris(m_block, method=method, k=k, eps=eps, sigma=sigma)

            f_corners = np.empty(shape=(top_k, 2), dtype=np.int32)
            for max_idx in range(top_k):
                idx = np.argmax(f_harris)
                idx_tuple = np.unravel_index(idx, shape=f_harris.shape)
                f_corners[max_idx] = np.array(idx_tuple)
                rr, cc = disk(f_corners[max_idx], radius=min_dist, shape=f_harris.shape)
                f_harris[rr, cc] = np.NINF

            m_corners = np.empty(shape=(top_k, 2), dtype=np.int32)
            for max_idx in range(top_k):
                idx = np.argmax(m_harris)
                idx_tuple = np.unravel_index(idx, shape=m_harris.shape)
                m_corners[max_idx] = np.array(idx_tuple)
                rr, cc = disk(m_corners[max_idx], radius=min_dist, shape=m_harris.shape)
                m_harris[rr, cc] = np.NINF

            f_harris_kp = [cv2.KeyPoint(x, y, sift_kp_size) for [y, x] in f_corners.tolist()]
            m_harris_kp = [cv2.KeyPoint(x, y, sift_kp_size) for [y, x] in m_corners.tolist()]

            f_8bit_block = f_8bit[i: row_end, j: col_end]
            m_8bit_block = m_8bit[i: row_end, j: col_end]

            # sift = cv2.xfeatures2d.SIFT_create()
            sift = cv2.SIFT_create()  # version: 4.5.5.62
            f_sift_kp, f_des = sift.compute(f_8bit_block, f_harris_kp)
            m_sift_kp, m_des = sift.compute(m_8bit_block, m_harris_kp)

            matcher = cv2.BFMatcher()
            raw_matches = matcher.knnMatch(f_des, m_des, k=2)
            good_matches = []
            for ei, pair in enumerate(raw_matches):
                try:
                    m1, m2 = pair
                    if m1.distance < match_ratio * m2.distance:
                        good_matches.append([m1])
                except ValueError:
                    pass
            for good_match in good_matches:
                idx1 = good_match[0].queryIdx
                idx2 = good_match[0].trainIdx
                f_pts.append(np.array([f_sift_kp[idx1].pt[1], f_sift_kp[idx1].pt[0]]) + np.array([i, j]))
                m_pts.append(np.array([m_sift_kp[idx2].pt[1], m_sift_kp[idx2].pt[0]]) + np.array([i, j]))

    f_pts = np.array(f_pts)
    m_pts = np.array(m_pts)

    return f_pts, m_pts


# def detect_block_harris_and_match1(f_gray, m_gray, f_img, m_img, n_rows=10, n_cols=10, top_k=50, method="k", k=0.05,
#                                    eps=1e-6,
#                                    sigma=1,
#                                    min_dist=10,
#                                    sift_kp_size=9, match_ratio=0.7):
#     f_8bit = img_to_8bit(f_img)
#     m_8bit = img_to_8bit(m_img)
#     f_corners = []
#     m_corners = []
#     f_pts = []
#     m_pts = []
#     row_step = f_gray.shape[0] // n_rows
#     col_step = f_gray.shape[1] // n_cols
#     for i in range(0, f_gray.shape[0], row_step):
#         for j in range(0, f_gray.shape[1], col_step):
#             row_end = i + row_step if i + row_step < f_gray.shape[0] else f_gray.shape[0]
#             col_end = j + col_step if j + col_step < f_gray.shape[1] else f_gray.shape[1]
#
#             f_block = f_gray[i:row_end, j:col_end]
#             f_harris = corner_harris(f_block, method=method, k=k, eps=eps, sigma=sigma)
#
#             m_block = m_gray[i:row_end, j:col_end]
#             m_harris = corner_harris(m_block, method=method, k=k, eps=eps, sigma=sigma)
#
#             for max_idx in range(top_k):
#                 idx = np.argmax(f_harris)
#                 idx_tuple = np.unravel_index(idx, shape=f_harris.shape)
#                 f_corners.append(np.array(idx_tuple) + np.array([i, j]))
#                 rr, cc = disk(f_corners[max_idx], radius=min_dist, shape=f_harris.shape)
#                 f_harris[rr, cc] = np.NINF
#
#             for max_idx in range(top_k):
#                 idx = np.argmax(m_harris)
#                 idx_tuple = np.unravel_index(idx, shape=m_harris.shape)
#                 m_corners.append(np.array(idx_tuple) + np.array([i, j]))
#                 rr, cc = disk(m_corners[max_idx], radius=min_dist, shape=m_harris.shape)
#                 m_harris[rr, cc] = np.NINF
#
#     print(len(f_corners))
#
#     f_harris_kp = [cv2.KeyPoint(x, y, sift_kp_size) for [y, x] in f_corners]
#     m_harris_kp = [cv2.KeyPoint(x, y, sift_kp_size) for [y, x] in m_corners]
#
#     sift = cv2.xfeatures2d.SIFT_create()
#     # sift = cv2.SIFT_create()  # version: 4.5.5.62
#     f_sift_kp, f_des = sift.compute(f_8bit, f_harris_kp)
#     m_sift_kp, m_des = sift.compute(m_8bit, m_harris_kp)
#
#     matcher = cv2.BFMatcher()
#     raw_matches = matcher.knnMatch(f_des, m_des, k=2)
#     good_matches = []
#     for m1, m2 in raw_matches:
#         if m1.distance < match_ratio * m2.distance:
#             good_matches.append([m1])
#
#     for good_match in good_matches:
#         idx1 = good_match[0].queryIdx
#         idx2 = good_match[0].trainIdx
#         f_pts.append(np.array([f_sift_kp[idx1].pt[1], f_sift_kp[idx1].pt[0]]))
#         m_pts.append(np.array([m_sift_kp[idx2].pt[1], m_sift_kp[idx2].pt[0]]))
#
#     f_pts = np.array(f_pts)
#     m_pts = np.array(m_pts)
#
#     return f_pts, m_pts
