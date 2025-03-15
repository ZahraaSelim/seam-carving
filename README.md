# Seam Carving

Python implementation of **Seam Carving** for CSE429 - Computer Vision Course. 

## Usage

You can run the script using the following arguments:
- -in / --input_image: Required. Path to the input image that you want to resize.
- -wf / --width_factor: Required. Factor by which to reduce the image's width. This should be an integer.
- -hf / --height_factor: Required. Factor by which to reduce the image's height. This should be an integer.
- -out / --output_dir: Required. The directory where the resized image, seams visualization, and energy map will be saved.

#### Example
``` 
python seam_carving.py -in './path/to/input.jpg' -out './path/to/output' -wf 2 -hf 1
```

## Time Tracking
```
python seam_carving.py -in './mountain/input.jpg' -out './mountain' -wf 2 -hf 1
```
time: 6.43 seconds

```
python seam_carving.py -in './castle/input.jpg' -out './castle' -wf 2 -hf 1
```
time: 53.74 seconds
