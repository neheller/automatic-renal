# Copyright 2021 Nicholas Heller. All Rights Reserved.
#
# Licensed under the MIT License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" A collection of functions which are useful for getting the necessary
information from the volume in order to compute nephrometry metrics """

from pathlib import Path

import numpy as np
import pydicom
from scipy.signal import convolve2d
from scipy.ndimage.measurements import label
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform
from pyfastnns import NNS
import time
import cv2


def get_centroid(volume):
    coordinates = np.transpose(np.array(np.nonzero(volume)))
    centroid = np.mean(coordinates, axis=0)
    return centroid


def _blur_thresh(vol):
    kernel = np.ones((3,3))/9.0
    ret = np.zeros(np.shape(vol), dtype=np.float32)
    for i in range(vol.shape[0]):
        ret[i] = convolve2d(
            vol[i], kernel, mode="same", boundary="fill", fillvalue=0
        )
    return ret


def _get_distance(c1, c2, x_width=1, y_width=1, z_width=1):
    return np.linalg.norm(
        np.multiply(c1 - c2, np.array((x_width, y_width, z_width))), ord=2
    )


def distance_between_regions(first_coordinates, second_coordinates):
    nns = NNS(first_coordinates)
    _, distance = nns.search(second_coordinates)
    min_distance = np.min(distance)
    return min_distance


def nearest_pair(first_coordinates, second_coordinates):
    nns = NNS(first_coordinates)
    pts, distances = nns.search(second_coordinates)
    min_distance_idx = np.argmin(distances)
    
    sp = second_coordinates[min_distance_idx]
    fp = first_coordinates[pts[min_distance_idx]]

    return fp, sp


def furthest_pair_distance(coordinates):
    coordinates = np.array(coordinates).T
    D = pdist(coordinates)
    return np.nanmax(D)


def get_nearest_rim_point(region_boundaries, pixel_width, slice_thickness):
    # Get coordinates of collecting system voxels
    rim_bin = np.equal(region_boundaries, 5).astype(np.int32)
    rim_coordinates = np.transpose(np.array(np.nonzero(rim_bin)))
    if rim_coordinates.shape[0] == 0:
        raise ValueError("Renal rim could not be identified")

    # Get coordinates of tumor voxels
    tumor_bin = np.equal(region_boundaries, 2).astype(np.int32)
    tumor_coordinates = np.transpose(np.array(np.nonzero(tumor_bin)))

    # Scale coordinates such that they correspond to the real world (mm)
    multiplier = np.array(
        [[slice_thickness, pixel_width, pixel_width]]
    ).astype(np.float32)
    rim_coordinates = np.multiply(rim_coordinates, multiplier)
    tumor_coordinates = np.multiply(tumor_coordinates, multiplier)

    nearest_pt, _ = nearest_pair(rim_coordinates, tumor_coordinates)
    return np.divide(nearest_pt, multiplier[0])


def get_distance_to_collecting_system(region_boundaries, pixel_width, 
    slice_thickness):
    # Get coordinates of collecting system voxels
    ucs_bin = np.equal(region_boundaries, 4).astype(np.int32)
    ucs_coordinates = np.transpose(np.array(np.nonzero(ucs_bin)))
    if ucs_coordinates.shape[0] == 0:
        return get_distance_to_sinus(
            region_boundaries, pixel_width, slice_thickness
        )
        # raise ValueError("Collecting system could not be identified")

    # Get coordinates of tumor voxels
    tumor_bin = np.equal(region_boundaries, 2).astype(np.int32)
    tumor_coordinates = np.transpose(np.array(np.nonzero(tumor_bin)))

    # Scale coordinates such that they correspond to the real world (mm)
    ucs_coordinates = np.multiply(
        ucs_coordinates, 
        np.array([[slice_thickness, pixel_width, pixel_width]])
    )
    tumor_coordinates = np.multiply(
        tumor_coordinates, 
        np.array([[slice_thickness, pixel_width, pixel_width]])
    )

    # Find nearest point between the two (quickly pls)
    min_distance = distance_between_regions(
        ucs_coordinates, tumor_coordinates
    )

    return min_distance


def get_distance_to_sinus(region_boundaries, pixel_width, 
    slice_thickness):
    # Get coordinates of collecting system voxels
    sinus_bin = np.equal(region_boundaries, 3).astype(np.int32)
    sinus_coordinates = np.array(np.nonzero(sinus_bin), dtype=np.float32).T
    if sinus_coordinates.shape[0] == 0:
        return np.inf
        # raise ValueError("Sinus could not be identified")

    # Get coordinates of tumor voxels
    tumor_bin = np.equal(region_boundaries, 2).astype(np.int32)
    tumor_coordinates = np.array(np.nonzero(tumor_bin), dtype=np.float32).T

    # Scale coordinates such that they correspond to the real world (mm)
    multiplier = np.array(
        [[slice_thickness, pixel_width, pixel_width]]
    ).astype(np.float32)

    tumor_coordinates = np.multiply(tumor_coordinates, multiplier) 

    sinus_coordinates = np.multiply(sinus_coordinates, multiplier)

    # Find nearest point between the two (quickly pls)
    min_distance = distance_between_regions(
        sinus_coordinates, tumor_coordinates
    )

    return min_distance



def prep_seg_shape(seg):
    """ Make sure segmentation is of the shape (slices, height, width) """
    if len(seg.shape) > 3:
        return np.reshape(seg, [seg.shape[0], seg.shape[1], seg.shape[2]])
    return seg


def get_pixel_width(dicom_directory):
    """ Returns the distance between adjacent pixel centers in millimeters 
    
    Needs a Path object where the volume dicoms live
    """
    for p in dicom_directory.glob("*"):
        try:
            dcm = pydicom.dcmread(str(p))
            return float(dcm[0x0028, 0x0030].value[0])
        except:
            continue
    raise IOError(
        "Unable to get a pixel spacing value for this directory: {0}".format(
            str(dicom_directory)
        )
    )
    return None


def get_slice_thickness(dicom_directory):
    """ Returns the distance between adjacent slices in millimeters
    
    Needs a Path object where the volume dicoms live
    """
    for p in dicom_directory.glob("*"):
        try:
            dcm = pydicom.dcmread(str(p))
            return float(dcm[0x0018, 0x0050].value)
        except:
            continue
    raise IOError("Unable to get a slices thickness value for this directory")
    return None


def load_volume(dicom_path, plat_id=None):
    if plat_id is not None:
        pth = Path(
            "/home/helle246/data/umnkcid/intermediate/volumes/{}.npy".format(
                plat_id
            )
        )
        if pth.exists():
            print("loading volume from {}".format(str(pth)))
            return np.load(str(pth))
    dcms = [pydicom.dcmread(str(slc)) for slc in dicom_path.glob("*")]
    instance_nums = [int(dcm[0x20,0x13].value) for dcm in dcms]
    spatial_shape = dcms[0].pixel_array.shape
    ret = np.zeros((len(dcms), spatial_shape[0], spatial_shape[1]))
    for i, ind in enumerate(np.argsort(instance_nums).tolist()):
        dcm = dcms[ind]
        data = dcm.pixel_array
        try:
            slope = float(dcm[0x28, 0x1053].value)
        except KeyError:
            slope = 1.0
        try:
            intercept = float(dcm[0x28, 0x1052].value)
        except KeyError:
            intercept = -1024.0 - data[0,0]

        ret[i] = slope*data + intercept

    return ret


def get_interior_seg_boundaries(seg):
    conv_kernel = np.ones((3,3), dtype=np.int32)
    ret = np.zeros(seg.shape, dtype=np.int32)
    for i in range(ret.shape[0]):
        for v in np.unique(seg[i]).tolist():
            if v != 0:
                bin_arr = np.zeros(seg[i].shape, dtype=np.int32)
                bin_arr[seg[i] == v] = 1
                conv = convolve2d(
                    bin_arr, conv_kernel, 
                    mode="same", boundary="fill", fillvalue=0
                )
                bin_arr = np.logical_and(
                    np.greater(bin_arr, 0),
                    np.less(conv, 9)
                )
                ret[i] = ret[i] + v*bin_arr

    return ret


def get_affected_kidney_subregions(seg, vol):
    # Get affected region, set seg to zero elsewhere
    components, _ = label(seg, structure=np.ones((3,3,3)))
    tumor_pixel_components = components[seg == 2]
    try:
        affected_component_ind = mode(tumor_pixel_components, axis=None)[0][0]
    except IndexError:
        print("Warning: could not identify tumor subregion")
        return None
    affected_seg = np.where(
        np.equal(components, affected_component_ind),
        seg, np.zeros(np.shape(seg), dtype=seg.dtype)
    )

    # Get outer boundary of affected region
    affected_region = np.greater(
        affected_seg, 0.5
    ).astype(seg.dtype)
    affected_interior = get_interior_seg_boundaries(affected_region)

    # Get sinus by blurring volume and finding kidney pixels below the 
    # threshold
    conv_kernel = np.ones((3,3), dtype=np.float32)/9
    blurred_volume = np.zeros(np.shape(vol))
    for i in range(vol.shape[0]):
        blurred_volume[i] = convolve2d(
            vol[i], conv_kernel, 
            mode='same', boundary='fill', fillvalue=vol[0,0,0]
        )
    sinus = np.where(
        np.logical_and(
            np.logical_and(
                np.less(blurred_volume, -30),
                np.greater(affected_seg, 0)
            ),
            np.less(affected_interior, 0.5)
        ),
        np.ones(np.shape(seg), dtype=seg.dtype),
        np.zeros(np.shape(seg), dtype=seg.dtype)
    )
    grown_sinus = sinus.copy()
    big_conv_kernel = np.ones((15,15), dtype=np.int32)
    for i in range(grown_sinus.shape[0]):
        grown_sinus[i] = np.where(
            np.greater(
                convolve2d(grown_sinus[i], big_conv_kernel, mode='same'), 0
            ),
            np.ones(np.shape(grown_sinus[i]), dtype=seg.dtype),
            np.zeros(np.shape(grown_sinus[i]), dtype=seg.dtype)
        )
    # Set sinus equal to largest connectect sinus component
    components, num_components = label(grown_sinus, structure=np.ones((3,3,3)))
    try:
        largest_component = mode(components[components != 0], axis=None)[0][0]
    except IndexError:
        largest_component = -1
    sinus = np.logical_and(
        np.equal(components, largest_component),
        np.equal(sinus, 1)
    ).astype(seg.dtype)
    ucs = np.zeros(np.shape(sinus), dtype=seg.dtype)
    for i in range(sinus.shape[0]):
        if 1 not in sinus[i]:
            continue
        # Compute binary image of convex hull of sinus
        contours, _ = cv2.findContours(
            255*sinus[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        fincont = contours[0]
        for cont in contours[1:]:
            fincont = np.concatenate((fincont, cont), axis=0)
        hull = cv2.convexHull(fincont)
        cv2.fillConvexPoly(ucs[i], hull, color=1)

        # Everything labeled kidney but not sinus in this is ucs
        ucs[i] = np.logical_and(
            np.logical_and(
                np.less(sinus[i], 1),
                np.greater(affected_seg[i], 0)
            ),
            np.greater(ucs[i], 0)
        ).astype(seg.dtype)

    # Get rim
    rim = np.logical_and(
        np.greater(affected_interior, 0),
        np.logical_and(
            np.less(sinus, 1),
            np.less(ucs, 1)
        )
    ).astype(seg.dtype)


    subregions = np.greater(affected_seg, 0).astype(seg.dtype)
    subregions = subregions + 2*sinus + 3*ucs + 4*rim
    subregions = np.where(np.equal(affected_seg, 2), affected_seg, subregions)
    return subregions

def _get_plane_center(seg_slice):
    """ Get the centroid pixel of a binary 2d slice of the segmentation """
    # Get center of mass in y-direction
    y_margin = np.sum(seg_slice, axis=1)
    y_range = np.arange(0, y_margin.shape[0])
    y_center = np.sum(np.multiply(y_margin, y_range))/np.sum(y_margin)
    # Get center of mass in x-direction
    x_margin = np.sum(seg_slice, axis=0)
    x_range = np.arange(0, x_margin.shape[0])
    x_center = np.sum(np.multiply(x_margin, x_range))/np.sum(x_margin)
    return int(y_center), int(x_center)


def get_tumor_center(seg):
    """ Get the centroid of the tumor """
    seg = prep_seg_shape(seg)
    binseg = np.equal(seg, 2).astype(np.int32)
    sums = np.sum(binseg, axis=(1,2))
    z_center = np.argmax(sums)
    y_center, x_center = _get_plane_center(binseg[z_center])
    return np.array((z_center, y_center, x_center), dtype=np.int32)


def count_tumor_voxels_by_type(tum_slc, kid_thresh_slc):
    # Set OR of all convex hulls to zeros. Will add to this over time
    convex_or = np.zeros(np.shape(tum_slc), dtype=np.uint8)

    # Get contours of kidney thresholded image
    contours, _ = cv2.findContours(
        kid_thresh_slc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Iterate over connected components and add convex hull to OR
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(convex_or, hull, color=1)

    # Count voxels of each type
    endophytic_count = np.sum(
        np.logical_and(
            np.equal(tum_slc, 1),
            np.equal(convex_or, 1)
        ).astype(np.int32)
    )
    exophytic_count = np.sum(tum_slc) - endophytic_count

    return exophytic_count, endophytic_count


def get_max_tumor_radius(label_slice):
    """ Get the maximum radius of the tumor in pixels """
    # Optimally takes boundaries
    binseg = np.equal(label_slice, 2).astype(np.uint8)
    return furthest_pair_distance(np.nonzero(binseg))/2



def get_polar_line_indicies(subregions):
    hilum_bin = np.logical_or(
        np.equal(subregions, 3),
        np.equal(subregions, 4)
    ).astype(np.int32)

    idx_1 = -1
    idx_2 = -1
    for i, slc in enumerate(hilum_bin):
        if not 3 in subregions[i] and 4 not in subregions[i]:
            continue 
        hilum_exterior_edge = np.logical_and(
            np.greater(
                convolve2d(
                    slc, np.ones((3,3), dtype=np.int32), mode='same'
                ),
                0
            ),
            np.equal(subregions[i], 0)
        ).astype(np.int32)
        if 1 in hilum_exterior_edge:
            if idx_1 == -1:
                idx_1 = i
            else:
                idx_2 = i

    # print(idx_1, idx_2)

    return (idx_1, idx_2)