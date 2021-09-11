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
"""A command line interface for computing the RENAL score given paths to input folders with an input image and segmentation. The result is written to a json file in the provided output folder."""
import json
from pathlib import Path
import argparse

import SimpleITK as sitk
import numpy as np
import nibabel as nib

from utils import get_affected_kidney_subregions, get_interior_seg_boundaries
from renal import get_R, get_E, get_N, get_A, get_L


INPUT_IMG_DIR = "/input/images/ct/"
INPUT_SEG_DIR = "/input/images/kidney-tumor-and-cyst/"
OUTPUT_DIR = "/output/"

IO_DICT = {
  ".nii.gz": "NiftiImageIO",
  ".mha": "MetaImageIO"
}


def _load_medical_image_as_nifti(parent, img_type="image"):
  pths = [x for x in parent.glob("*")]
  if len(pths) > 1:
    raise ValueError("Expected a single {img_type} file to process, found", str(pths))
  if len(pths) == 0:
    raise ValueError("Expected a single {img_type} file to process, found none.")

  img_pth = pths[0]

  ext = img_pth.name[len(img_pth.name.split(".")[0]):]

  if ext not in IO_DICT:
    raise ValueError("Input {img_type} files must be of one of the following forms:", [x for x in IO_DICT], "-- found", ext)

  reader = sitk.ImageFileReader()
  reader.SetImageIO(IO_DICT[ext])
  reader.SetFileName(str(img_pth))
  image = reader.Execute()

  nda = np.moveaxis(sitk.GetArrayFromImage(image), -1, 0)
 
  spacing = image.GetSpacing()
  affine = np.array(
      [[0.0, 0.0, -1*spacing[2], 0.0],
      [0.0, -1*spacing[1], 0.0, 0.0],
      [-1*spacing[0], 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]]
  )

  return nib.Nifti1Image(nda, affine)


def load_image(parent):
  return _load_medical_image_as_nifti(parent, img_type="image")


def load_segmentation(parent):
  return _load_medical_image_as_nifti(parent, img_type="segmentation")

def clip_imaging(npimg, npseg):
  znz = np.multiply(
    np.greater(np.apply_over_axes(np.sum, npseg, (1,2)).flatten(), 0), 
    np.arange(0, npseg.shape[0])
  )
  ynz = np.multiply(np.greater(np.apply_over_axes(np.sum, npseg, (0,2)).flatten(),0), np.arange(0, npseg.shape[1]))
  xnz = np.multiply(np.greater(np.apply_over_axes(np.sum, npseg, (0,1)).flatten(),0), np.arange(0, npseg.shape[2]))

  zmin = max(np.argmin(znz)-10, 0)
  zmax = min(np.argmax(znz)+10, npseg.shape[0])
  ymin = max(np.argmin(ynz)-10, 0)
  ymax = min(np.argmax(ynz)+10, npseg.shape[1])
  xmin = max(np.argmin(xnz)-10, 0)
  xmax = min(np.argmax(xnz)+10, npseg.shape[2])

  return npimg[zmin:zmax,ymin:ymax,xmin:xmax], npseg[zmin:zmax,ymin:ymax,xmin:xmax]


def compute_renal(img, seg):
  npimg = np.asanyarray(img.dataobj)
  npseg = np.asanyarray(seg.dataobj)

  npimg, npseg = clip_imaging(npimg, npseg)

  subregions = get_affected_kidney_subregions(npimg, npseg)
  subregion_boundaries = get_interior_seg_boundaries(subregions)
  slice_thickness = np.abs(img.affine[0,2])
  pixel_width = np.abs(img.affine[1,1])

  print("Getting R")
  R, _ = get_R(subregions, subregion_boundaries, slice_thickness, pixel_width)
  print("Getting E")
  E, _ = get_E(subregions, subregion_boundaries, slice_thickness, pixel_width)
  print("Getting N")
  N, _ = get_N(subregions, subregion_boundaries, slice_thickness, pixel_width)
  print("Getting A")
  A, _ = get_A(subregions, subregion_boundaries, slice_thickness, pixel_width)
  print("Getting L")
  L, _ = get_L(subregions, subregion_boundaries, slice_thickness, pixel_width)

  ret = {
    "R": R,
    "E": E,
    "N": N,
    "A": A,
    "L": L
  }

  ret["RENAL"] = str(ret["R"] + ret["E"] + ret["N"] + ret["L"]) + ret["A"]

  return ret


def main(args):
  # Check i/o paths
  img_pth = Path(args.img_dir).resolve(strict=True)
  seg_pth = Path(args.seg_dir).resolve(strict=True)
  out_pth = Path(args.out_dir).resolve(strict=True)

  # Load Image
  img = load_image(img_pth)

  # Load Segmentation
  seg = load_segmentation(seg_pth)

  # Compute RENAL Score
  renal_results = compute_renal(img, seg)

  # Save Output
  with (out_pth / "results.json").open('w') as f:
    f.write(json.dumps(renal_results, indent=2))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Provide custom input and output locations, if desired.")
  parser.add_argument("--input-img-dir", dest="img_dir", default=INPUT_IMG_DIR)
  parser.add_argument("--input-seg-dir", dest="seg_dir", default=INPUT_SEG_DIR)
  parser.add_argument("--output-dir", dest="out_dir", default=OUTPUT_DIR)
  main(parser.parse_args())
