## Note
This is a forked version of the [refer api](https://github.com/lichengunc/refer). This distribution is compatible with Python3 and contains the new SUN-SPOT dataset.

## Setup
Run "make" before using the code.
It will generate ``_mask.c`` and ``_mask.so`` in ``external/`` folder.
These mask-related codes are copied from mscoco [API](https://github.com/pdollar/coco).

## Download:
If you want to use the refcoco, refcoco+, refcocog, or refclef datasets follow the [download instructions](https://github.com/lichengunc/refer/tree/master/data) from the original repository and add the files to the data folder. 

## Prepare Images:
Add "SUNRGBD" into the ``data/images`` folder, which can be from [sunrgbd](http://rgbd.cs.princeton.edu)

## How to use

The "refer.py" is able to load all 4 datasets from the original repository [refer api](https://github.com/lichengunc/refer) as well as the new SUNRGBD dataset. 

```bash
# locate your own data_root, and choose the dataset_splitBy you want to use
refer = REFER(data_root, dataset='sunspot',  splitBy='boulder') #The new dataset!
refer = REFER(data_root, dataset='refclef',  splitBy='unc')
refer = REFER(data_root, dataset='refclef',  splitBy='berkeley') # 2 train and 1 test images missed
refer = REFER(data_root, dataset='refcoco',  splitBy='unc')
refer = REFER(data_root, dataset='refcoco',  splitBy='google')
refer = REFER(data_root, dataset='refcoco+', splitBy='unc')
refer = REFER(data_root, dataset='refcocog', splitBy='google')   # test split not released yet
refer = REFER(data_root, dataset='refcocog', splitBy='umd')      # Recommended, including train/val/test
```


<!-- refs(dataset).p contains list of refs, where each ref is
{ref_id, ann_id, category_id, file_name, image_id, sent_ids, sentences}
ignore filename

Each sentences is a list of sent
{arw, sent, sent_id, tokens}
 -->
