# Extrinsic Calibration

Hand-eye calibration. Three images is enough to get good result.

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

## Project Dependencies
- opencv
- apriltag (pip install apriltag)
- open3d (with read_point_cloud_with_nan)
- numpy
- json
- transforms3d

## References
1. [Least-Squares Rigid Motion Using SVD](https://igl.ethz.ch/projects/ARAP/svd_rot.pdf)
2. [Quaternion Averaging](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf)
3. [Unified Temporal and Spatial Calibration for Multi-Sensor Systems](https://furgalep.github.io/bib/furgale_iros13.pdf)