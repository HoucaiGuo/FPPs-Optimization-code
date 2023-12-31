import numpy as np
from g_im_stats import img_corrcoef, img_mutual_info
from g_im_display import match_hist
from g_im_io import normalize_image, check_dim
from skimage.metrics import structural_similarity


def delete_point(pt_set, del_idx):
    """
    Delete a point at the given index from a point set.

    Parameters
    ----------
    pt_set : array_like
        A set of points.
    del_idx : int
        Index of the point to delete.

    Returns
    -------
    del_pnt_set : array_like
        Point set after deletion.
    """
    if del_idx == 0:
        return pt_set[1:, :]
    elif del_idx == pt_set.shape[0] - 1:
        return pt_set[:del_idx, :]
    else:
        left = pt_set[:del_idx, :]
        right = pt_set[del_idx + 1:, :]
        return np.concatenate([left, right], axis=0)


def delete_attribute(attr, del_idx):
    if del_idx == 0:
        return attr[1:]
    elif del_idx == attr.shape[0] - 1:
        return attr[:del_idx]
    else:
        left = attr[:del_idx]
        right = attr[del_idx + 1:]
        return np.concatenate([left, right], axis=0)


def similarity_measure(f_img, reg_img, criterion="zncc", mode="image"):
    """
    Calculate the similarity between two images.

    Parameters
    ----------
    f_img : array_like
        Fixed image, (H, W, C) shaped.
    reg_img : array_like
        Registered image.
    criterion : str, optional
        Similarity measure criterion to use. Default is "zncc".
        "zncc" : Zero mean Normalized Cross-Correlation.
        "mi" : Mutual Information.
        "nmi" : Normalized Mutual Information.
        "ssim" : Structural Similarity.
        "mad" : Mean of Absolute Difference.
    mode : str, optional
        Mode when calculating the similarity. Use "image" to take the whole image as a variable, use "pixel" to treat
            each vector of pixel value along the channel axis as a variable. Default is "image".

    Returns
    -------
    similarity : array_like
        The similarity between two images, which is a scalar when the mode is "image" and an (H, W, 1) shaped array when
            the mode is "pixel".
    """
    if criterion == "zncc":
        similarity = img_corrcoef(reg_img, f_img, mode)
    elif criterion == "mi":
        similarity = img_mutual_info(reg_img, f_img, mode, normalize=False)
    elif criterion == "nmi":
        similarity = img_mutual_info(reg_img, f_img, mode, normalize=True)
    elif criterion == "ssim":
        reg_img.data[reg_img.mask] = -1
        f_img.data[f_img.mask] = -1
        similarity = structural_similarity(reg_img, f_img, gaussian_weights=True, sigma=1.5,
                                           use_sample_covariance=True, multichannel=True)
    elif criterion == "mad":
        similarity = np.mean(np.abs(reg_img - f_img))
    elif criterion == "ssd":
        similarity = np.sum(np.square(reg_img - f_img))
    else:
        similarity = img_corrcoef(reg_img, f_img, mode)

    return similarity


# def checker_board(img1, img2, n_rows=10, n_cols=10, hist_match=False):
#     cb_img = np.ma.empty_like(img1)
#     if hist_match:
#         img2 = match_hist(img2, img1)
#
#     row_step = img1.shape[0] // n_rows
#     col_step = img1.shape[1] // n_cols
#     block_idx = 1
#     for i in range(0, img1.shape[0], row_step):
#         for j in range(0, img1.shape[1], col_step):
#             row_end = i + row_step if i + row_step < img1.shape[0] else img1.shape[0]
#             col_end = j + col_step if j + col_step < img1.shape[1] else img1.shape[1]
#
#             if block_idx % 2 != 0:
#                 cb_img[i:row_end, j:col_end, :] = img1[i:row_end, j:col_end, :]
#             else:
#                 cb_img[i:row_end, j:col_end, :] = img2[i:row_end, j:col_end, :]
#
#     cb_img.data[cb_img.mask] = np.nan
#
#     return cb_img


def checker_board(img1, img2, step, hist_match=False):
    """
    Make a checkerboard image.

    Parameters
    ----------
    img1 :
    img2 :
    step :
    hist_match : bool


    Returns
    -------

    """
    img1 = check_dim(img1)
    img2 = check_dim(img2)

    cb_img = np.ma.empty_like(img1)
    if hist_match:
        img2 = match_hist(img2, img1)

    for i in range(0, cb_img.shape[0], step):
        for j in range(0, cb_img.shape[1], step):
            if ((i + j) / step) % 2 == 0:
                cb_img[i:i + step + 1, j:j + step + 1, :] = img1[i:i + step + 1, j:j + step + 1, :]
            else:
                cb_img[i:i + step + 1, j:j + step + 1, :] = img2[i:i + step + 1, j:j + step + 1, :]

    cb_img.data[cb_img.mask] = np.nan

    return cb_img


def img_to_8bit(img):
    img_norm = normalize_image(img)
    img_8bit = np.ma.stack([np.uint8(img_norm[:, :, i] * 255) for i in range(img_norm.shape[2])], axis=2)

    return img_8bit
