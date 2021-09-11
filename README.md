# Automatic Renal

This repository contains an algorithm which computes the RENAL nephrometry score based on a CT image that has been segmented in the way that was done for the KiTS19 Challenge.

This algorithm has been deployed via grand-challenge. You can run it through the web-based interface [here](https://grand-challenge.org/algorithms/kits19-based-automatic-renal-scoring/experiments/create/flex/).

## Example Output

```json
{
  "R": 3,
  "E": 2,
  "N": 1,
  "A": "x",
  "L": 3,
  "RENAL": "9x"
}
```

## Usage

The algorithm requires three folders to be created: One that contains a single CT image file in either `.nii.gz` or `.mha` format (referred to as `LOCAL_IMAGE_DIR`), another that contains a corresponding segmentation file in the same format (referred to as `LOCAL_SEGMENTATION_DIR`), and finally an output folder in which to save the resulting `.json` file (referred to as `LOCAL_OUTPUT_DIR`).

### Docker

To build and run via Docker

```bash
docker build automatic-renal -t automatic_renal
```

```bash
docker run -v LOCAL_IMAGE_DIR:/input/images/ct/ -v LOCAL_SEGMENTATION_DIR:/input/images/kidney-tumor-and-cyst/ -v LOCAL_OUTPUT_DIR:/output/ automatic_renal
```

### Within Python

If you'd prefer to run the code without Docker, you must enter a python environment and install the following dependencies

- Numpy
- SimpleITK
- Nibabel
- Pydicom
- Scipy
- OpenCV

Then run the following command:

```bash
python3 automatic-renal/cli.py --input-img-dir LOCAL_IMAGE_DIR --input-seg-dir LOCAL_SEGMENTATION_DIR --output-dir LOCAL_OUTPUT_DIR
```
