This directory should contain the following data:
```
$DATA_PATH
├── images
│   ├── SUNRGBD
├── sunspot
│   ├── instances.json
│   ├── refs(boulder).p
```

Note, each detections/xxx.json contains 
``{'dets': ['box': {x, y, w, h}, 'image_id', 'object_id', 'score']}``. The ``object_id`` and ``score`` might be missing, depending on what proposal/detection technique we are using.

## Download

1.Make a folder named as "images".
2.Add "SUNRGBD" into "images/". 
Download SUNRGBD from [http://rgbd.cs.princeton.edu](http://rgbd.cs.princeton.edu)

If you want to use the refcoco, refcoco+, refcocog, or refclef datasets follow the [download instructions](https://github.com/lichengunc/refer/tree/master/data) from the original repository and add the files to the data folder. 

Note: These formats are the same as the refcoco, refcocog, refcoco+, and refclef datasets from [refer api](https://github.com/lichengunc/refer) and the all datasets can be loaded with either API