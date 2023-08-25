from g_block_harris import *
from g_im_io import *
from g_im_display import *
from skimage.color import rgb2gray
from scipy.spatial import Delaunay
from datetime import datetime
from pl_optimization import OutlierRemoval, PointsOptimization
from skimage.measure import ransac
from skimage.transform import AffineTransform

"""
Parameters Setting
"""
# if True, a SIFT-based global Affine Transformation is applied first,
# then follows the proposed Optimized Piecewise Linear Transformation
is_large_deformation = False
# if show the intermediate results
show = True
# n and k for blockwise Harris Corner detection
k = 30
n_rows = 9
n_cols = 9
# the threshold of TIN-TARR, default "1.05"
tin_tarr_threshold = 1.05
# similarity measurement, default "ssim"
similarity_criterion = "ssim"

# intput path
f_img_path = r"data/f_img.tif"
m_img_path = r"data/m_img.tif"
# output path
f_initial_pts_path = r"data/f_pts.csv"
m_initial_pts_path = r"data/m_pts.csv"
f_inliers_path = r"data/f_inliers.csv"
m_inliers_path = r"data/m_inliers.csv"
f_optim_pts_path = f"data/{similarity_criterion}_f_optim_pts.csv"
m_optim_pts_path = f"data/{similarity_criterion}_m_optim_pts.csv"
optim_pl_img_path = f"data/{similarity_criterion}_optim_pl_img.tif"

line_color = "#f9ca24"
f_pt_color = "#d63031"
m_pt_color = "#0984e3"

if __name__ == "__main__":
    # region read, normalize, stretch and color composite image
    f_img, prof = read_image(f_img_path)
    print(prof)
    f_img = f_img.astype(np.float32)
    m_img = read_image(m_img_path)[0].astype(np.float32)
    f_norm = normalize_image(f_img)
    m_norm = normalize_image(m_img)

    f_norm = linear_pct_stretch(f_norm, 5)
    m_match = match_hist(m_norm, f_norm)
    # endregion

    # region pre-register
    if is_large_deformation:
        from skimage.transform import estimate_transform, warp

        sift = cv2.xfeatures2d.SIFT_create()
        f_8bit = img_to_8bit(f_img)
        m_8bit = img_to_8bit(m_img)
        f_sift_kp, f_des = sift.detectAndCompute(f_8bit, None)
        m_sift_kp, m_des = sift.detectAndCompute(m_8bit, None)

        f_sift_pts = []
        m_sift_pts = []
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(f_des, m_des, k=2)
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < 0.7 * m2.distance:
                good_matches.append([m1])

        for good_match in good_matches:
            idx1 = good_match[0].queryIdx
            idx2 = good_match[0].trainIdx
            f_sift_pts.append(np.array([f_sift_kp[idx1].pt[1], f_sift_kp[idx1].pt[0]]))
            m_sift_pts.append(np.array([m_sift_kp[idx2].pt[1], m_sift_kp[idx2].pt[0]]))

        f_sift_pts = np.array(f_sift_pts)
        m_sift_pts = np.array(m_sift_pts)

        src = np.flip(f_sift_pts, axis=1)
        dst = np.flip(m_sift_pts, axis=1)
        model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                       residual_threshold=1, max_trials=100)
        f_sift_inliers = f_sift_pts[inliers]
        m_sift_inliers = m_sift_pts[inliers]

        tform = estimate_transform("affine", src=np.flip(f_sift_inliers, axis=1), dst=np.flip(m_sift_inliers, axis=1))
        reg_img = warp(m_img, tform, output_shape=f_img.shape, order=3, cval=-1)
        reg_img = np.ma.array(data=reg_img, mask=(reg_img == -1))
        reg_norm = normalize_image(reg_img)
        reg_match = match_hist(reg_norm, f_norm)

        print("Pre-registered image.")

        if show:
            # region show initial points and TIN
            fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
            axes[0].imshow(f_norm[:, :, 0], cmap="gray")
            axes[1].imshow(m_match[:, :, 0], cmap="gray")
            axes[2].imshow(reg_match[:, :, 0], cmap="gray")
            axes[0].set_title("fixed image")
            axes[1].set_title("moving image")
            axes[2].set_title("pre-registered image")
            for i in range(f_sift_inliers.shape[0]):
                axes[0].plot(f_sift_inliers[i, 1], f_sift_inliers[i, 0], color=f_pt_color, marker=".", markersize=6)
                axes[1].plot(m_sift_inliers[i, 1], m_sift_inliers[i, 0], color=m_pt_color, marker=".", markersize=6)
            plt.show()
            # endregion

        m_match = reg_match
    # endregion

    # region generate initial FPPs using block-Harris and SIFT
    time0 = datetime.now()
    f_pts = []
    m_pts = []
    for band_idx in range(f_img.shape[2]):
        f_band_pts, m_band_pts = detect_block_harris_and_match1(f_norm[:, :, band_idx], m_match[:, :, band_idx],
                                                                f_norm, m_match,
                                                                n_rows=n_rows, n_cols=n_cols, top_k=k,
                                                                method="eps", sigma=1)
        f_pts.append(f_band_pts)
        m_pts.append(m_band_pts)

    f_pts = np.concatenate(f_pts, axis=0)
    m_pts = np.concatenate(m_pts, axis=0)

    f_pts, indices = np.unique(f_pts, return_index=True, axis=0)
    m_pts = m_pts[indices]
    m_pts, indices = np.unique(m_pts, return_index=True, axis=0)
    f_pts = f_pts[indices]

    time1 = datetime.now()
    time_span = time1 - time0
    print(f"Detected and matched {f_pts.shape[0]} FPPs. Used {time_span.total_seconds():.3f} seconds.")
    # endregion

    if show:
        # region show initial points and TIN
        tin = Delaunay(f_pts)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].imshow(f_norm[:, :, 0], cmap="gray")
        axes[1].imshow(m_match[:, :, 0], cmap="gray")
        axes[0].set_title("fixed image")
        axes[1].set_title("moving image")
        axes[0].triplot(f_pts[:, 1], f_pts[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        axes[1].triplot(m_pts[:, 1], m_pts[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        for i in range(f_pts.shape[0]):
            axes[0].plot(f_pts[i, 1], f_pts[i, 0], color=f_pt_color, marker=".", markersize=6)
            axes[1].plot(m_pts[i, 1], m_pts[i, 0], color=m_pt_color, marker=".", markersize=6)
        plt.show()
        # endregion

    # region save initial FPPs
    f_file = open(f_initial_pts_path, mode="x")
    m_file = open(m_initial_pts_path, mode="x")
    for i in range(f_pts.shape[0]):
        f_file.write(f"{f_pts[i, 0]},{f_pts[i, 1]}\n")
        m_file.write(f"{m_pts[i, 0]},{m_pts[i, 1]}\n")
    f_file.close()
    m_file.close()
    # endregion

    # region outlier removal
    print("Start outlier removal.")
    f_bdy_pts = np.array([[0, 0], [0, f_img.shape[1] - 1], [f_img.shape[0] - 1, 0],
                          [f_img.shape[0] - 1, f_img.shape[1] - 1]])
    m_bdy_pts = np.array([[0, 0], [0, m_img.shape[1] - 1], [m_img.shape[0] - 1, 0],
                          [m_img.shape[0] - 1, m_img.shape[1] - 1]])
    outlier_removal = OutlierRemoval(f_pts, m_pts, f_bdy_pts, m_bdy_pts, threshold=tin_tarr_threshold, del_bdy=True)
    time0 = datetime.now()
    outlier_removal.remove_wrong_match()
    time1 = datetime.now()
    time_span = time1 - time0
    f_inliers = outlier_removal.f_pts
    m_inliers = outlier_removal.m_pts
    print(f"Finished the removal of outliers, {f_inliers.shape[0]} remain. "
          f"Used {time_span.total_seconds():.3f} seconds.")
    # endregion

    if show:
        # region show inliers and TIN
        tin = Delaunay(f_inliers)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].imshow(f_norm[:, :, 0], cmap="gray")
        axes[1].imshow(m_match[:, :, 0], cmap="gray")
        axes[0].set_title("fixed image")
        axes[1].set_title("moving image")
        axes[0].triplot(f_inliers[:, 1], f_inliers[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        axes[1].triplot(m_inliers[:, 1], m_inliers[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        for i in range(f_inliers.shape[0]):
            axes[0].plot(f_inliers[i, 1], f_inliers[i, 0], color=f_pt_color, marker=".", markersize=6)
            axes[1].plot(m_inliers[i, 1], m_inliers[i, 0], color=m_pt_color, marker=".", markersize=6)
        plt.show()
        # endregion

    # region save inliers
    f_file = open(f_inliers_path, mode="x")
    m_file = open(m_inliers_path, mode="x")
    for i in range(f_inliers.shape[0]):
        f_file.write(f"{f_inliers[i, 0]},{f_inliers[i, 1]}\n")
        m_file.write(f"{m_inliers[i, 0]},{m_inliers[i, 1]}\n")
    f_file.close()
    m_file.close()
    # endregion

    # region points optimization, image registration
    print("Start FPPs optimization.")
    time0 = datetime.now()
    plt_optim = PointsOptimization(f_img, m_img, f_inliers, m_inliers, criterion=similarity_criterion)
    plt_optim.optimize()
    time1 = datetime.now()
    time_span = time1 - time0
    f_optim = plt_optim.f_pts
    m_optim = plt_optim.m_pts
    print(f"Finished the optimization of CPs, {f_optim.shape[0]} remain. Used {time_span.total_seconds():.3f} seconds.")
    # endregion

    if show:
        # region show optimal points and TIN
        tin = Delaunay(f_optim)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[0].imshow(f_norm[:, :, 0], cmap="gray")
        axes[1].imshow(m_match[:, :, 0], cmap="gray")
        axes[0].set_title("fixed image")
        axes[1].set_title("moving image")
        axes[0].triplot(f_optim[:, 1], f_optim[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        axes[1].triplot(m_optim[:, 1], m_optim[:, 0], tin.simplices, color=line_color, linewidth=1.5)
        for i in range(f_optim.shape[0]):
            axes[0].plot(f_optim[i, 1], f_optim[i, 0], color=f_pt_color, marker=".", markersize=6)
            axes[1].plot(m_optim[i, 1], m_optim[i, 0], color=m_pt_color, marker=".", markersize=6)
        plt.show()
        # endregion

    # region save optimized points and perform optimized PLT
    f_file = open(f_optim_pts_path, mode="x")
    m_file = open(m_optim_pts_path, mode="x")
    for i in range(f_optim.shape[0]):
        f_file.write(f"{f_optim[i, 0]},{f_optim[i, 1]}\n")
        m_file.write(f"{m_optim[i, 0]},{m_optim[i, 1]}\n")
    f_file.close()
    m_file.close()

    optim_img = plt_optim.transform()
    save_image(optim_img, optim_pl_img_path, prof)
    # endregion
