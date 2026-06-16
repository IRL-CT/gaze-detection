# Gaze Detection Models, Features

## Total Features List
- A. Nose-Eye Offset
    - Ratio of *horizontal* distance from eye center to nose and eye distance.

- B. Ear-Nose Offset
    - Ratio between *perpendicular* distance from earline to nose and earline length. 
        - earline = [`Left Ear -- Right Ear`]

- C. Nose-Eyeline Ratio
    - Ratio between *Pythagorean* distances of [`Left Eye -- Nose`] and [`Right Eye -- Nose`].

- D. Number of Ears [0~2]
    - Number of ears confidently detected by YOLO.


## BEST MODEL: Gradient Boosting 2param
- VA: 76%

- Features: A, B
    - Most consistent, does not flicker btwn APPROACH/DNI as much.
<!-- - Features: Nose-Eye, Ear-Nose Offsets. -->


## GB/RF NENl: Nose, Ear Offsets + Nose-Eyeline Ratio
- VA: 70%, 74% (GB, RF)

- Features: A, B, C
<!-- - Features: Nose-Eye, Ear-Nose Offsets, Nose-Eyeline Ratio. -->

## RF v77: Random Forest
- VA: 77%

- Features: A, B
    - Highest accuracy, but in practice worse; flickers often.
<!-- - Features: Nose-Eye, Ear-Nose Offsets. -->
