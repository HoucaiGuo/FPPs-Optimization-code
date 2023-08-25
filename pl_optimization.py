import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp
from functions import delete_point, delete_attribute, similarity_measure
from g_im_io import normalize_image
from g_im_display import color_composite
from datetime import datetime
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

shapely.speedups.disable()

np.set_printoptions(suppress=True)


class TIN_TARR:
    def __init__(self, f_pts, m_pts):
        self.f_pts = f_pts
        self.m_pts = m_pts
        self.tin = Delaunay(self.f_pts)

    def __single_triangle_tar_ratio(self, tri):
        f_tri = self.f_pts[tri]
        m_tri = self.m_pts[tri]
        # formula (1)
        f_tar = np.abs(f_tri[0, 0] * f_tri[1, 1] + f_tri[1, 0] * f_tri[2, 1] + f_tri[2, 0] * f_tri[0, 1] -
                       f_tri[2, 0] * f_tri[1, 1] - f_tri[0, 0] * f_tri[2, 1] - f_tri[1, 0] * f_tri[0, 1]) / 2
        # # formula (2)
        # affine = estimate_transform("affine", src=np.flip(f_tri, axis=1), dst=np.flip(m_tri, axis=1))
        # # formula (3)
        # m_tar = (affine.params[0, 0] * affine.params[1, 1] - affine.params[0, 1] * affine.params[1, 0]) * f_tar

        """
        to minimize the computational efficiency, the m_tar is directly calculated using the coordinates 
        of the m_tri's vertices
        """
        m_tar = np.abs(m_tri[0, 0] * m_tri[1, 1] + m_tri[1, 0] * m_tri[2, 1] + m_tri[2, 0] * m_tri[0, 1] -
                       m_tri[2, 0] * m_tri[1, 1] - m_tri[0, 0] * m_tri[2, 1] - m_tri[1, 0] * m_tri[0, 1]) / 2

        if m_tar != 0:
            # formula (4)
            tar_ratio = f_tar / m_tar

            # formula (5)
            if tar_ratio < 1:
                tar_ratio = 1 / tar_ratio

        else:
            # tar_ratio = 999
            tar_ratio = np.PINF

        return tar_ratio

    def tin_tar_ratio(self, pt_idx):
        triangles = self.tin.simplices[np.any(np.isin(self.tin.simplices, pt_idx), axis=1)]
        tar_ratios = np.empty((triangles.shape[0],), dtype=np.float64)
        for tri_idx in range(triangles.shape[0]):
            tar_ratios[tri_idx] = self.__single_triangle_tar_ratio(triangles[tri_idx])
        # formula (6)
        tin_tar_ratio = np.mean(tar_ratios)

        return tin_tar_ratio

    def tin_tar_ratios(self):
        tin_tar_ratios = np.empty((self.f_pts.shape[0],), dtype=np.float64)
        for pt_idx in range(self.f_pts.shape[0]):
            tin_tar_ratios[pt_idx] = self.tin_tar_ratio(pt_idx)

        return tin_tar_ratios


class OutlierRemoval:
    def __init__(self, f_pts, m_pts, f_bdy_pts, m_bdy_pts, threshold=1.1, del_bdy=True):
        self.f_pts = np.concatenate([f_pts, f_bdy_pts], axis=0)
        self.m_pts = np.concatenate([m_pts, m_bdy_pts], axis=0)
        self.f_bdy_pts = f_bdy_pts
        self.m_bdy_pts = m_bdy_pts
        self.threshold = threshold

        # if delete the boundary points after outlier removal
        self.del_bdy = del_bdy

        self.tin = Delaunay(self.f_pts)
        self.convex_hull = ConvexHull(self.f_pts)
        self.outside_ia = np.empty(shape=(self.f_pts.shape[0],), dtype=np.bool)
        self.tin_tar_ratios = TIN_TARR(self.f_pts, self.m_pts).tin_tar_ratios()

        # set the boundary points' TIN-TARR value to -1, they will not be deleted in the automatic process
        self.tin_tar_ratios[-self.f_bdy_pts.shape[0]:] = -1

        # record the deleted points
        self.del_f_pts = []
        self.del_m_pts = []

        # calculate and record TIN-TARRs of the deleted points
        self.del_tin_tarrs = []

        # indices of delete points' neighbors
        self.del_nei_indices = []

    def __point_outside_ia(self, pt_idx):
        """
        Determine if a moving point is outside its "influence area".
        Topological consistency check.

        Parameters
        ----------
        pt_idx : int
            Index of the point.

        Returns
        -------
        outside_ia : bool
            True if this point is outside its influence area, else False.
        """
        if pt_idx in self.convex_hull.vertices:
            outside_ia = False
        else:
            triangles = self.tin.simplices[np.any(np.isin(self.tin.simplices, pt_idx), axis=1)]
            pnt = Point(self.m_pts[pt_idx, 0], self.m_pts[pt_idx, 1])
            plgs = [Polygon(list(tuple(map(tuple, self.m_pts[tri])))) for tri in triangles]
            plg = unary_union(plgs)
            # if the m_pt is inside its Influence Area
            outside_ia = False if plg.contains(pnt) else True

        return outside_ia

    def __del_tin_tarr(self, pt_idx):
        """
        Calculate a deleted point's TIN-TARR value by inserting it into the TIN. The insertion of this point would
        change its neighbors' TIN-TARR values, so the del_tin_tarr is defined as:
                            del_tin_tarr = max(nei_tin_tarrs ∪ del_tin_tarr)

        Parameters
        ----------
        pt_idx : int
            Point index.

        Returns
        -------
        del_tin_tarr : float
            TIN-TARR value of the deleted point.
        nei_indices : numpy.array
            Indices of its neighbors in the TIN.
        """
        f_pts = np.concatenate([self.f_pts, np.expand_dims(self.del_f_pts[pt_idx], axis=0)])
        m_pts = np.concatenate([self.m_pts, np.expand_dims(self.del_m_pts[pt_idx], axis=0)])
        tin = Delaunay(f_pts)

        # test if the deleted point is the first type of outlier
        triangles = tin.simplices[np.any(np.isin(tin.simplices, f_pts.shape[0] - 1), axis=1)]
        pnt = Point(m_pts[f_pts.shape[0] - 1, 0], m_pts[f_pts.shape[0] - 1, 1])
        plgs = [Polygon(list(tuple(map(tuple, m_pts[tri])))) for tri in triangles]
        plg = unary_union(plgs)
        if not plg.contains(pnt):
            return 999, []
        else:
            tt = TIN_TARR(f_pts, m_pts)
            # the inserted point has a index of "f_pts.shape[0] - 1"
            tin_tarr = tt.tin_tar_ratio(f_pts.shape[0] - 1)

            # get indices of its neighbors, then calculate their TIN-TARR value
            # after the insertion of this deleted point
            indptr, indices = tin.vertex_neighbor_vertices
            nei_indices = indices[indptr[f_pts.shape[0] - 1]:indptr[f_pts.shape[0] - 1 + 1]]
            nei_tin_tarrs = np.empty(shape=(nei_indices.shape[0],), dtype=np.float64)
            for i in range(nei_indices.shape[0]):
                nei_tin_tarrs[i] = tt.tin_tar_ratio(nei_indices[i])

            nei_max = np.max(nei_tin_tarrs)

            del_tin_tarr = nei_max if nei_max > tin_tarr else tin_tarr

            return del_tin_tarr, nei_indices

    def __insert(self):
        """
        Iteratively insert the deleted points into the TIN if they have a TIN-TARR value lower than the threshold.
        """
        # calculate the TIN-TARR values of the deleted points
        for pt_idx in range(len(self.del_f_pts)):
            max_tin_tarr, del_nei_indices = self.__del_tin_tarr(pt_idx)
            self.del_tin_tarrs.append(max_tin_tarr)
            self.del_nei_indices.append(del_nei_indices)

        while np.min(self.del_tin_tarrs) <= self.threshold:
            insert_idx = np.argmin(self.del_tin_tarrs)

            self.f_pts = np.concatenate([self.f_pts, np.expand_dims(self.del_f_pts[insert_idx], axis=0)])
            self.m_pts = np.concatenate([self.m_pts, np.expand_dims(self.del_m_pts[insert_idx], axis=0)])
            self.tin = Delaunay(self.f_pts)
            self.convex_hull = ConvexHull(self.f_pts)
            self.tin_tar_ratios = np.concatenate([self.tin_tar_ratios,
                                                  np.expand_dims(np.array(self.del_tin_tarrs[insert_idx]), axis=0)],
                                                 axis=0)
            self.del_f_pts.pop(insert_idx)
            self.del_m_pts.pop(insert_idx)
            self.del_tin_tarrs.pop(insert_idx)
            nei_indices = self.del_nei_indices[insert_idx]
            self.del_nei_indices.pop(insert_idx)

            # modify the TIN-TARR values of other deleted points who may be effected by the insertion
            for pt_idx in range(len(self.del_f_pts)):
                for nei_idx in nei_indices:
                    if nei_idx in self.del_nei_indices[pt_idx]:
                        max_tin_tarr, del_nei_indices = self.__del_tin_tarr(pt_idx)
                        self.del_tin_tarrs[pt_idx] = max_tin_tarr
                        self.del_nei_indices[pt_idx] = del_nei_indices

    def remove_wrong_match(self):
        """
        Remove the three types of wrongly matched point pairs respectively and iteratively.
        """
        # calculate the topological consistency
        for pt_idx in range(self.f_pts.shape[0]):
            self.outside_ia[pt_idx] = self.__point_outside_ia(pt_idx)

        # 1st type
        while True in self.outside_ia:
            # find point which should be deleted
            sort_indices = np.flip(np.argsort(self.tin_tar_ratios))

            for sort_idx in sort_indices:
                if self.outside_ia[sort_idx]:
                    del_idx = sort_idx
                    break

            indptr, indices = self.tin.vertex_neighbor_vertices
            nei_indices = indices[indptr[del_idx]:indptr[del_idx + 1]]
            for i in range(nei_indices.shape[0]):
                if nei_indices[i] > del_idx:
                    nei_indices[i] -= 1

            # delete this FPP and its attributes
            self.del_f_pts.append(self.f_pts[del_idx])
            self.del_m_pts.append(self.m_pts[del_idx])
            self.f_pts = delete_point(self.f_pts, del_idx)
            self.m_pts = delete_point(self.m_pts, del_idx)
            self.outside_ia = delete_attribute(self.outside_ia, del_idx)
            self.tin_tar_ratios = delete_attribute(self.tin_tar_ratios, del_idx)
            self.tin = Delaunay(self.f_pts)
            self.convex_hull = ConvexHull(self.f_pts)

            for pt_idx in nei_indices:
                # do not modify the manually selected boundary points' TIN-TARR and topological consistency
                if pt_idx not in range(self.f_pts.shape[0] - self.f_bdy_pts.shape[0], self.f_pts.shape[0]):
                    self.outside_ia[pt_idx] = self.__point_outside_ia(pt_idx)
                    self.tin_tar_ratios[pt_idx] = TIN_TARR(self.f_pts, self.m_pts).tin_tar_ratio(pt_idx)

        # 2nd and 3rd type
        while np.max(self.tin_tar_ratios) > self.threshold:
            del_idx = np.argmax(self.tin_tar_ratios)

            indptr, indices = self.tin.vertex_neighbor_vertices
            nei_indices = indices[indptr[del_idx]:indptr[del_idx + 1]]
            for i in range(nei_indices.shape[0]):
                if nei_indices[i] > del_idx:
                    nei_indices[i] -= 1

            self.del_f_pts.append(self.f_pts[del_idx])
            self.del_m_pts.append(self.m_pts[del_idx])
            self.f_pts = delete_point(self.f_pts, del_idx)
            self.m_pts = delete_point(self.m_pts, del_idx)
            self.tin_tar_ratios = delete_attribute(self.tin_tar_ratios, del_idx)
            self.tin = Delaunay(self.f_pts)
            self.convex_hull = ConvexHull(self.f_pts)

            for pt_idx in nei_indices:
                # do not modify the manually selected boundary points' TIN-TARR
                if pt_idx not in range(self.f_pts.shape[0] - self.f_bdy_pts.shape[0], self.f_pts.shape[0]):
                    self.tin_tar_ratios[pt_idx] = TIN_TARR(self.f_pts, self.m_pts).tin_tar_ratio(pt_idx)

        # insert the wrongly deleted points
        self.__insert()

        # delete added boundary points
        if self.del_bdy:
            for i in range(self.f_bdy_pts.shape[0]):
                for j in range(self.f_pts.shape[0]):
                    if np.all(self.f_bdy_pts[i] == self.f_pts[j]):
                        self.f_pts = delete_point(self.f_pts, j)
                        self.m_pts = delete_point(self.m_pts, j)
                        break


class PointsOptimization:
    """
    Optimize the CP pairs used in PLT by deleting the CP pair with negative influence on its local accuracy,
    iteratively.

    The method consists of two main procedures:
        1. Calculate the Accuracy Difference (AD) of each CP pair by applying a specific similarity measurement. The
        Accuracy Difference is defined as "acc_del" minus "acc_use", where "acc_use" represent the similarity between
        the local patch in the fixed image and the local patch of the registered image (also called the accuracy) when
        using this CP pair to estimate a PLT; "acc_del" is the accuracy when discarding this CP pair to estimate a PLT.

        2. Delete CP pairs that have the maximum Accuracy Difference iteratively. For a positive AD value indicates this
        CP pair has a negative influence on the registration accuracy of the local image patch, so all of the retained
        CP pairs would have a positive impact on the registration accuracy.

    Parameters
    ----------
    f_img : numpy.array
        The fixed image, (H, W, C) shaped.
    m_img : numpy.array
        The moving image.
    f_pts : numpy.array
        CPs detected from the fixed image, (n, 2) shaped.
    m_pts : numpy.array
        CPs detected from the moving image.
    criterion : str, optional
        Similarity measure criterion to use. Default is "zncc".
        "zncc" : Zero mean Normalized Cross-Correlation.
        "mi" : Mutual Information.
        "nmi" : Normalized Mutual Information.
        "ssim" : Structural Similarity.
    order : int, optional
        The order of interpolation, default is 3. The order has to be in the range 0-5:
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic

    Attributes
    ----------
    f_img, m_img, f_pts, m_pts, criterion, order: same as Parameters.
    tin : scipy.Delaunay
        The triangulation of f_pts.
    convex_hull : scipy.ConvexHull
        The convex hull of f_pts.
    acc_diffs : numpy.array
        An array that records AD of each point.
    del_acc_diffs : list
        A list that records the maximum AD value in each iteration.
    del_f_pts : list
        List of deleted f_pt.
    del_m_pts : list
        List of deleted m_pt.
    """

    def __init__(self, f_img, m_img, f_pts, m_pts, criterion="zncc", order=3):
        self.f_img = f_img.astype(np.float32)
        self.m_img = m_img.astype(np.float32)
        self.f_pts = f_pts
        self.m_pts = m_pts
        self.criterion = criterion
        self.order = order
        self.tin = Delaunay(self.f_pts)
        self.convex_hull = ConvexHull(self.f_pts)
        self.acc_diffs = np.empty(shape=(self.f_pts.shape[0],), dtype=np.float64)
        self.del_acc_diffs = []
        self.del_f_pts = []
        self.del_m_pts = []

    def __influence_area_triangles(self, pt_idx):
        """
        Find triangles that form the "influence area" of a given point in the TIN. For a single point, the influence
        area is a polygon consists of its adjacent points.

        Parameters
        ----------
        pt_idx : int
            Index of the point.

        Returns
        -------
        triangles : array_like
            Triangles that make up the influence area, (n, 3) shaped, each record along the first axis contains 3
                indices of vertices that compose a triangle.
        """
        # use the nesting of np.any() and np.isin() to judge if the point is a vertex of a triangle(tin.simplex)
        triangles = self.tin.simplices[np.any(np.isin(self.tin.simplices, pt_idx), axis=1)]

        return triangles

    def __influence_area(self, pt_idx, nei_ia=True):
        """
        Extract fixed and moving images inside the influence area of a given point.

        Parameters
        ----------
        pt_idx : int
            Index of the point.
        nei_ia : bool, optional
            If add the neighbors points' influence area into this point's influence area.

        Returns
        -------
        f_ia_img : array_like
            The fixed image inside the influence area.
        m_ia_img : array_like
            The moving image inside the influence area.
        """
        ia_tris = self.__influence_area_triangles(pt_idx)

        if nei_ia:
            # indices of the adjacent points
            nbr_pts_idx = np.setdiff1d(np.unique(ia_tris), pt_idx)

            # its neighbors' influence area triangles
            for nbr_idx in nbr_pts_idx:
                ia_tris = np.concatenate([ia_tris, self.__influence_area_triangles(nbr_idx)])
            # every triangle in ia_tris is unique
            ia_tris = np.unique(ia_tris, axis=0)

        f_ia_mask = polygon2mask(self.f_img.shape,
                                 self.f_pts[ia_tris[0]])
        m_ia_mask = polygon2mask(self.m_img.shape,
                                 self.m_pts[ia_tris[0]])

        for i in range(1, ia_tris.shape[0]):
            f_tri_mask = polygon2mask(self.f_img.shape,
                                      self.f_pts[ia_tris[i]])
            f_ia_mask = np.bitwise_or(f_ia_mask, f_tri_mask)

            m_tri_mask = polygon2mask(self.m_img.shape,
                                      self.m_pts[ia_tris[i]])
            m_ia_mask = np.bitwise_or(m_ia_mask, m_tri_mask)

        # idx is the index of point inside the f_ia_pts
        pts_indices = np.unique(ia_tris)
        idx = np.where(pts_indices == pt_idx)[0][0]

        # coordinates of the triangles' vertices, (p, 2)shaped
        f_ia_pts = self.f_pts[pts_indices]
        m_ia_pts = self.m_pts[pts_indices]

        # extract sub-image by envelope of the influence area
        f_row_min = np.min(f_ia_pts[:, 0]).astype(np.int32)
        f_row_max = np.max(f_ia_pts[:, 0]).astype(np.int32)
        f_col_min = np.min(f_ia_pts[:, 1]).astype(np.int32)
        f_col_max = np.max(f_ia_pts[:, 1]).astype(np.int32)
        f_ia_img = self.f_img[f_row_min:f_row_max + 1, f_col_min:f_col_max + 1, :].copy()
        f_ia_mask = f_ia_mask[f_row_min:f_row_max + 1, f_col_min:f_col_max + 1, :]
        m_row_min = np.min(m_ia_pts[:, 0]).astype(np.int32)
        m_row_max = np.max(m_ia_pts[:, 0]).astype(np.int32)
        m_col_min = np.min(m_ia_pts[:, 1]).astype(np.int32)
        m_col_max = np.max(m_ia_pts[:, 1]).astype(np.int32)
        m_ia_img = self.m_img[m_row_min:m_row_max + 1, m_col_min:m_col_max + 1, :].copy()
        m_ia_mask = m_ia_mask[m_row_min:m_row_max + 1, m_col_min:m_col_max + 1, :]

        # transfer points' coordinate from image to sub-image, (m, 3, 2) shaped
        f_ia_pts = f_ia_pts - np.array([f_row_min, f_col_min])
        m_ia_pts = m_ia_pts - np.array([m_row_min, m_col_min])

        return f_ia_img, m_ia_img, f_ia_mask, m_ia_mask, f_ia_pts, m_ia_pts, idx

    def __single_point_test(self, pt_idx, nei_ia=True, show=False):
        """
        Calculate the AD for a single CP pair.
        """
        # test if its point is on the boundary
        is_bdy = np.any(np.in1d(self.convex_hull.vertices, pt_idx))
        if is_bdy:
            acc_diff = np.NINF
        else:
            # calculate this point's acc_diff
            f_ia_img, m_ia_img, f_ia_mask, m_ia_mask, f_ia_pts, m_ia_pts, idx = self.__influence_area(pt_idx, nei_ia)

            f_ia_img = np.ma.array(data=f_ia_img, mask=~f_ia_mask)
            f_ia_img.data[f_ia_img.mask] = np.nan

            tform_use = estimate_transform("piecewise-affine", src=np.flip(f_ia_pts, axis=1),
                                           dst=np.flip(m_ia_pts, axis=1))
            reg_img_use = warp(m_ia_img, tform_use, output_shape=f_ia_img.shape, order=self.order)

            reg_img_use = np.ma.array(data=reg_img_use, mask=~f_ia_mask)
            reg_img_use.data[reg_img_use.mask] = np.nan

            # print(f"{f_ia_img.shape} {np.count_nonzero(f_ia_img.mask)}\n"
            #       f"{reg_img_use.shape} {np.count_nonzero(reg_img_use.mask)}")

            acc_use = similarity_measure(f_ia_img, reg_img_use, criterion=self.criterion, mode="image")

            f_ia_pts_del = delete_point(f_ia_pts, idx)
            m_ia_pts_del = delete_point(m_ia_pts, idx)
            tform_del = estimate_transform("piecewise-affine",
                                           src=np.flip(f_ia_pts_del, axis=1),
                                           dst=np.flip(m_ia_pts_del, axis=1))
            reg_img_del = warp(m_ia_img, tform_del, output_shape=f_ia_img.shape, order=self.order)
            reg_img_del = np.ma.array(data=reg_img_del, mask=~f_ia_mask)
            reg_img_del.data[reg_img_del.mask] = np.nan

            acc_del = similarity_measure(f_ia_img, reg_img_del, criterion=self.criterion, mode="image")

            acc_diff = acc_del - acc_use
            if show:
                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

                f_ia_rgb = color_composite(normalize_image(f_ia_img), [2, 1, 0])
                f_ia_rgb.data[f_ia_rgb.mask] = 1

                m_ia_rgb = color_composite(normalize_image(np.ma.array(data=m_ia_img, mask=~m_ia_mask)), [2, 1, 0])
                m_ia_rgb.data[m_ia_rgb.mask] = 1

                reg_use_rgb = color_composite(normalize_image(reg_img_use), [2, 1, 0])
                reg_use_rgb.data[reg_use_rgb.mask] = 1

                reg_del_rgb = color_composite(normalize_image(reg_img_del), [2, 1, 0])
                reg_del_rgb.data[reg_del_rgb.mask] = 1

                f_tin = Delaunay(f_ia_pts)
                # print(f_tin.simplices)
                m_tin = Delaunay(m_ia_pts)
                f_del_tin = Delaunay(f_ia_pts_del)
                # print(f_del_tin.simplices)
                axes[0, 0].imshow(f_ia_rgb)
                axes[0, 0].set_title("fixed")
                axes[0, 0].triplot(f_ia_pts[:, 1], f_ia_pts[:, 0], f_tin.simplices, color="gold", linewidth=1)
                axes[0, 0].set_xlim(-40, f_ia_rgb.shape[1] + 40)
                axes[0, 0].set_ylim(-40, f_ia_rgb.shape[0] + 40)

                axes[0, 1].imshow(m_ia_rgb)
                axes[0, 1].set_title("moving")
                axes[0, 1].triplot(m_ia_pts[:, 1], m_ia_pts[:, 0], f_tin.simplices, color="gold", linewidth=1)

                axes[1, 0].imshow(reg_use_rgb, origin="upper")
                axes[1, 0].set_title(f"use, "
                                     f"acc: {similarity_measure(f_ia_img, reg_img_use, criterion=self.criterion, mode='image'):.5f}")
                axes[1, 0].triplot(f_ia_pts[:, 1], f_ia_pts[:, 0], f_tin.simplices, color="gold", linewidth=1)
                axes[1, 1].imshow(reg_del_rgb, origin="upper")
                axes[1, 1].set_title(f"discard, "
                                     f"acc: {similarity_measure(f_ia_img, reg_img_del, criterion=self.criterion, mode='image'):.5f}")
                axes[1, 1].triplot(f_ia_pts_del[:, 1], f_ia_pts_del[:, 0], f_del_tin.simplices, color="gold",
                                   linewidth=1)

                for i in range(f_ia_pts.shape[0]):
                    axes[0, 0].plot(f_ia_pts[i, 1], f_ia_pts[i, 0], color="gold", marker=".", markersize=3)
                    axes[0, 0].annotate(f"{i}", xy=(f_ia_pts[i, 1], f_ia_pts[i, 0]), color="red", fontsize=12)
                    axes[0, 1].plot(m_ia_pts[i, 1], m_ia_pts[i, 0], color="gold", marker=".", markersize=3)
                    axes[0, 1].annotate(f"{i}", xy=(m_ia_pts[i, 1], m_ia_pts[i, 0]), color="red", fontsize=12)
                    axes[1, 0].plot(f_ia_pts[i, 1], f_ia_pts[i, 0], color="gold", marker=".", markersize=3)
                    axes[1, 0].annotate(f"{i}", xy=(f_ia_pts[i, 1], f_ia_pts[i, 0]), color="red", fontsize=12)
                for i in range(f_ia_pts_del.shape[0]):
                    axes[1, 1].plot(f_ia_pts_del[i, 1], f_ia_pts_del[i, 0], color="gold", marker=".", markersize=3)
                    axes[1, 1].annotate(f"{i}", xy=(f_ia_pts_del[i, 1], f_ia_pts_del[i, 0]), color="red", fontsize=12)
                print(idx)

                plt.show()

        return acc_diff

    def __single_point_optimize(self, pt_idx):
        """
        Delete a single FPP, then update its neighbors' loss.
        """
        # print(f"{pt_idx} should be deleted, "
        #       f"f_pt: {self.f_pts[pt_idx]}, m_pt: {self.m_pts[pt_idx]}, acc_diff: {self.acc_diffs[pt_idx]}")

        indptr, indices = self.tin.vertex_neighbor_vertices
        nei_indices = indices[indptr[pt_idx]:indptr[pt_idx + 1]]
        # print(f"\tIt's neighbors: {nei_indices}")

        self.f_pts = delete_point(self.f_pts, pt_idx)
        self.m_pts = delete_point(self.m_pts, pt_idx)
        self.acc_diffs = delete_attribute(self.acc_diffs, pt_idx)
        self.tin = Delaunay(self.f_pts)
        self.convex_hull = ConvexHull(self.f_pts)

        for i in range(nei_indices.shape[0]):
            if nei_indices[i] > pt_idx:
                nei_indices[i] -= 1

        for nei_idx in nei_indices:
            original_acc_diff = self.acc_diffs[nei_idx]
            acc_diff = self.__single_point_test(nei_idx)
            # print(f"\t\tidx: {nei_idx}, original acc_diff: {original_acc_diff}, new acc_diff: {acc_diff}")
            self.acc_diffs[nei_idx] = acc_diff

    def optimize(self):
        """
        Optimize the FPPs.
        """
        # print("Start calculating points' accuracy difference!")
        time0 = datetime.now()
        for pt_idx in range(self.f_pts.shape[0]):
            self.acc_diffs[pt_idx] = self.__single_point_test(pt_idx)
            # print(f"idx: {pt_idx}, f_pt: {self.f_pts[pt_idx]}, m_pt: {self.m_pts[pt_idx]}, "
            #       f"acc_diff: {self.acc_diffs[pt_idx]}")
        time1 = datetime.now()
        time_span0 = time1 - time0

        # print("Start optimizing!")
        time0 = datetime.now()
        max_diff_idx = np.argsort(self.acc_diffs)[-1]
        max_diff = self.acc_diffs[max_diff_idx]
        while max_diff > 0:
            self.del_acc_diffs.append(max_diff)
            self.del_f_pts.append(self.f_pts[max_diff_idx])
            self.del_m_pts.append(self.m_pts[max_diff_idx])

            self.__single_point_optimize(max_diff_idx)

            max_diff_idx = np.argsort(self.acc_diffs)[-1]
            max_diff = self.acc_diffs[max_diff_idx]

        time1 = datetime.now()
        time_span1 = time1 - time0
        # print(f"Finished the optimization of CPs, used {time_span0.total_seconds():.2f} seconds to calculate the ADs, "
        #       f"{time_span1.total_seconds():.2f} to optimize.")

    def transform(self):
        """
        Estimate and perform the PLT.
        """
        tform = estimate_transform("piecewise-affine", src=np.flip(self.f_pts, axis=1), dst=np.flip(self.m_pts, axis=1))
        reg_img = warp(self.m_img, tform, output_shape=self.f_img.shape, order=self.order)
        mask = polygon2mask(self.f_img.shape, self.f_pts[self.convex_hull.vertices])
        reg_img = np.ma.array(data=reg_img, mask=~mask)
        reg_img.data[reg_img.mask] = np.nan

        return reg_img

    def transform_coordinates(self, f_acc_pts, m_acc_pts):
        """
        For each fixed CP in f_acc_pts, find which triangle it lays in, then use this triangle's affine transform
        parameters to transform its corresponding moving CP.

        Parameters
        ----------
        f_acc_pts : numpy.array
            FPPs extracted from the fixed image to calculate the RMSE.
        m_acc_pts : numpy.array
            Corresponding moving FPPs.

        Returns
        -------
        tri_idx : numpy.array
            Indices of triangles containing the fixed CPs.
        tformed_pts : numpy.array
            The transformed CPs.
        """
        tri_idx = np.empty(f_acc_pts.shape[0], dtype=np.int32)
        tformed_pnts = np.empty(f_acc_pts.shape, dtype=np.float32)
        for i in range(f_acc_pts.shape[0]):
            f_pnt = f_acc_pts[i]
            m_pnt = m_acc_pts[i]
            tformed_pnt = np.empty(m_pnt.shape, dtype=np.float32)
            idx = self.tin.find_simplex(f_pnt)
            tri_idx[i] = idx
            # estimate inverse transform
            f_tri_pnts = self.f_pts[self.tin.simplices[idx]]
            m_tri_pnts = self.m_pts[self.tin.simplices[idx]]
            tri_affine = estimate_transform("affine", src=m_tri_pnts, dst=f_tri_pnts)
            mat = tri_affine.params
            tformed_pnt[0] = mat[0, 0] * m_pnt[0] + mat[0, 1] * m_pnt[1] + mat[0, 2]
            tformed_pnt[1] = mat[1, 0] * m_pnt[0] + mat[1, 1] * m_pnt[1] + mat[1, 2]
            tformed_pnts[i] = tformed_pnt

        return tri_idx, tformed_pnts
