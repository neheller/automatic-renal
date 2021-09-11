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
""" A collection of functions which are useful for computing the renal score of 
a kidney tumor given semantic segmentation of the kidney and the tumor """

import numpy as np

from utils import get_centroid, get_tumor_center, get_max_tumor_radius, \
                   count_tumor_voxels_by_type, get_distance_to_collecting_system, \
                   get_polar_line_indicies


def get_R(subregions, subregion_boundaries, slice_thickness, pixel_width):
    tumor_center = get_tumor_center(subregions)
    max_radius_mm = get_max_tumor_radius(subregions[tumor_center[0]])
    diameter_mm = 2*max_radius_mm
    if diameter_mm >= 70:
        return 3, diameter_mm
    elif diameter_mm > 40:
        return 2, diameter_mm
    else:
        return 1, diameter_mm


def get_E(subregions, subregion_boundaries, slice_thickness, pixel_width):
    # Get kidney (not tumor) voxels as though it's the 2nd output of cv2 thresh
    kidney_no_tumor_thresh = 255*np.logical_or(
        np.equal(
            subregions, 1
        ),
        np.greater(
            subregions, 2.5
        )
    ).astype(np.uint8)
    # Get a 3d array with tumor as value 1 only 
    tumor_no_kidney_bin = np.equal(
        subregions, 2
    ).astype(np.uint8)

    tot_exophytic = 0
    tot_endophytic = 0
    for i in range(kidney_no_tumor_thresh.shape[0]):
        if np.sum(tumor_no_kidney_bin[i]) > 0:
            this_exophytic, this_endophytic = count_tumor_voxels_by_type(
                tumor_no_kidney_bin[i], kidney_no_tumor_thresh[i]
            )
            tot_exophytic = tot_exophytic + this_exophytic
            tot_endophytic = tot_endophytic + this_endophytic

    if tot_exophytic == 0:
        return 3, 1.0
    elif tot_endophytic > tot_exophytic:
        return 2, tot_endophytic/(tot_endophytic+tot_exophytic)
    else:
        return 1, tot_endophytic/(tot_endophytic+tot_exophytic)


def get_N(subregions, subregion_boundaries, slice_thickness, pixel_width):
    distance_mm = get_distance_to_collecting_system(
        subregion_boundaries, pixel_width, slice_thickness
    )
    if distance_mm >= 7:
        return 1, distance_mm
    elif distance_mm > 4:
        return 2, distance_mm
    else:
        return 3, distance_mm


def get_A(subregions, subregion_boundaries, slice_thickness, pixel_width):
    component_bin = np.greater(subregions, 0.5).astype(np.int32)
    component_centroid = get_centroid(component_bin)
    tumor_bin = np.equal(subregions, 2).astype(np.int32)
    tumor_centroid = get_centroid(tumor_bin)
    dst = (tumor_centroid[1] - component_centroid[1])*pixel_width

    # TODO neither?
    if np.abs(dst) < 10:
        return "x", dst
    elif tumor_centroid[1] < component_centroid[1]:
        return "a", dst
    else:
        return "p", dst


def get_L(subregions, subregion_boundaries, slice_thickness, pixel_width):
    # Get slices representing polar lines (interior of them)
    start, end = get_polar_line_indicies(subregions)

    # Define midline index/indices
    if (start + end)%2 == 0:
        midline = [(start + end)//2]
    else:
        midline = [(start + end - 1)//2, (start + end + 1)//2]

    # If tumor involves the midline, it is a three
    for m in midline:
        if 2 in subregions[m]:
            return 3, -1

    # Count total tumor pixels
    tumor_volume = np.sum(np.equal(subregions, 2).astype(np.int32))

    # Count tumor pixels between the polar lines
    count = 0
    for i in range(start, end+1):
        count = count + np.sum(np.equal(subregions[i], 2).astype(np.int32))

    # If more than halfway across the line, it's a 3
    # If 0 < t <= 0.5, then it's a 2, else it's a 1
    if count/tumor_volume > 0.5:
        return 3, count/tumor_volume
    if count/tumor_volume > 1e-4:
        return 2, count/tumor_volume
    else:
        return 1, count/tumor_volume

