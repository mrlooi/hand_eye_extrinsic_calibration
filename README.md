# Hand-eye Extrinsic Calibration

Hand-eye calibration. 6-8 images/pointclouds with a calibration board are enough to get good result.  
Currently supports asymmetric circles and april tags

## Examples
__Asymmetric circles:__ `python main_asymm_circle.py`  
![](./img/asymm_circle_example.png)

__April Tags__ (make sure there is only __one__ tag in the view! Multi tag is not supported): `python main_april_tag.py`  
![](./img/april_tag_example.png)


## Requirements
- numpy
- opencv
- apriltag (pip install apriltag)
- open3d (with read_point_cloud_with_nan)
- json
- transforms3d

## Algorithm Details
![](./img/hand-eye%20calibration.png)

## Results
### Before
![](./img/before_traj.png)
![](./img/before_xyz.png)
![](./img/before_rpy.png)

### After
![](./img/after_traj.png)
![](./img/after_xyz.png)
![](./img/after_rpy.png)

## References
1. [Least-Squares Rigid Motion Using SVD](https://igl.ethz.ch/projects/ARAP/svd_rot.pdf)
2. [Quaternion Averaging](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf)
3. [Unified Temporal and Spatial Calibration for Multi-Sensor Systems](https://furgalep.github.io/bib/furgale_iros13.pdf)