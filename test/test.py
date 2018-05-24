
from refer import REFER

import matplotlib.pyplot as plt
from pprint import pprint

if __name__ == '__main__':
    refer = REFER(dataset='sunspot', splitBy='boulder')
    ref_ids = refer.getRefIds()
    print(len(ref_ids))

    print(len(refer.Imgs))
    print(len(refer.imgToRefs))

    ref_ids = refer.getRefIds(split='train')
    print('There are %s training referred objects.' % len(ref_ids))

    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if len(ref['sentences']) < 2:
            continue

        pprint(ref)
        print('The label is %s.' % refer.Cats[ref['category_id']])
        plt.figure()
        refer.showRef(ref, seg_box='box')
        plt.show()

    # plt.figure()
    # refer.showMask(ref)
    # plt.show()