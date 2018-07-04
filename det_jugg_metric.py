import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="jugg_det toolkit",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iou', default=0.5, type=int, help="iou threshold")

    parser.add_argument(
        '--detected_json', help='model output file ', default=None, type=str)

    parser.add_argument(
        '--gt_json', help='ground truth json file, labelX export', default=None, type=str)
    parser.add_argument(
        '--output_log', help='output_log_path', type=str)
    args = parser.parse_args()
    return args


args = parse_args()


def get_all_bbox(gt_json, det_json):
    allBoundingBoxes = BoundingBoxes()
    det = json.load(open(det_json,'r'))
    for k, v in det.items():
        nameOfImage = k
        for obj in v:
            idClass = obj[5]  # class
            confidence = float(obj[4])  # confidence
            x1 = float(obj[0])
            y1 = float(obj[1])
            x2 = float(obj[2])
            y2 = float(obj[3])
            bb = BoundingBox(nameOfImage, idClass, x1, y1, x2, y2, bbType=BBType.Detected,
                             classConfidence=confidence, format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)

    gt = pd.read_json(gt_json, lines=True)
    for i in range(gt.shape[0]):
        url = gt.loc[i]['url']
        filename = url.split('/')[-1]
        if filename in imgs:
            if len(gt.loc[i]['label'][0]['data']) > 0:
                    for ix, data in enumerate(gt.loc[i]['label'][0]['data']):
                        x1, y1 = data['bbox'][0]
                        x2, y2 = data['bbox'][2]
                        idClass = data['class']
                        bb = BoundingBox(filename, idClass,x1,y1,x2,y2, bbType=BBType.GroundTruth,
                                         classConfidence=confidence, format=BBFormat.XYX2Y2)
                        allBoundingBoxes.addBoundingBox(bb)

    return allBoundingBoxes


if __name__=='__main__':
    boundBoxes = get_all_bbox(args.gt_json, args.detected_json)
    metricsPerClass = evaluator.GetPascalVOCMetrics(boundBoxes,
                                                    # Object containing all bounding boxes (ground truths and detections)
                                                    IOUThreshold=args.iou)  # IOU threshold
    with open(args.output_log, 'w') as f:
        print("Average precision values per class:")
        f.write("Average precision values per class:\n")
        for mc in metricsPerClass:
            # Get metric values per each class
            c = mc['class']
            precision = mc['precision']
            recall = mc['recall']
            average_precision = mc['AP']
            ipre = mc['interpolated precision']
            irec = mc['interpolated recall']
            # Print AP per class
            print('%s: %f' % (c, average_precision))
            f.write('%s: %f\n' % (c, average_precision))

