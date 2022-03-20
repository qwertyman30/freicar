"""
COCO-Style Evaluations
"""

import argparse
import torch
import yaml
from tqdm import tqdm
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes
from utils import postprocess, boolean_string
from dataloader.freicar_dataloader import FreiCarDataset
from model.efficientdet.dataset import collater
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

########################################################################
# Object Detection model evaluation script
# Modified by: Jannik Zuern (zuern@informatik.uni-freiburg.de)
########################################################################


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='freicar-detection', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--title', type=str, default="Precision-recall curve")
args = ap.parse_args()


compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device

project_name = args.project
weights_path = args.weights
title = args.title

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

threshold = 0.2
iou_threshold = 0.2


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou


def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax


def calculate_iou_for_boxes(rois, annot, ious):
    if rois.shape[0] == 0:
        if len(annot) == 1 and len(np.unique(annot[0].cpu().numpy())) == 1:
            return ious
        else:
            ##false negative
            for i in range(annot.shape[0]):
                ious.append(0)
            return ious
    else:
        # skip case of no bounding boxes in annotations (as mentioned in exercises)
        if len(annot) == 1 and len(np.unique(annot[0].cpu().numpy())) == 1:
            return ious
        else:
            # if
            for y in annot:
                max_iou = 0
                for prediction in rois:
                    iou = get_iou(prediction, y)
                    if iou > max_iou:
                        max_iou = iou
                ious.append(max_iou)
            return ious
        

if __name__ == '__main__':

    '''
    Note: 
    When calling the model forward function on an image, the model returns
    features, regression, classification and anchors.
    
    In order to obtain the final bounding boxes from these predictions, they need to be postprocessed
    (this performs score-filtering and non-maximum suppression)
    
    Thus, you should call
    

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold, nms_threshold)                  
    preds = preds[0]

    Now, the scores, class_indices and bounding boxes are saved as fields in the preds dict and can be used for subsequent evaluation.
    '''

    set_name = 'validation'

    freicar_dataset = FreiCarDataset(data_dir="./dataloader/data/",
                                     padding=(0, 0, 12, 12),
                                     split=set_name,
                                     load_real=False)
    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': 1}

    freicar_generator = DataLoader(freicar_dataset, **val_params)

    # instantiate model
    model = EfficientDetBackbone(compound_coef=compound_coef,
                                 num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']),
                                 scales=eval(params['anchors_scales']))
    
    # load model weights file from disk
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model = model.cuda()
    ious = []
    precision_recal_dict = {}
    results = []
    with torch.no_grad():
        for iter, data in enumerate(tqdm(freicar_generator)):

            #if iter == 100:
            #    break

            imgs = data['img'].cuda()
            annot = data['annot'].cuda()
            features, regression, classification, anchors = model(imgs)
            # cls_loss, reg_loss = criterion(classification, regression, anchors, annot)
            # loss = cls_loss.mean() + reg_loss.mean()

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()


            preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                                nms_threshold)[0]

            rois = preds['rois']
            scores = preds['scores']
            class_ids = preds['class_ids']

            mean_ious = {}
            ids = {}

            annot = annot.squeeze(0).cpu()

            probability_thresholds = np.linspace(0, 1, num=100)

            ious = calculate_iou_for_boxes(rois, annot, ious)


            for p in probability_thresholds:
                true_positive_tmp = 0
                false_positive_tmp = 0
                false_negative_tmp = 0
                preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, p,
                                    nms_threshold)[0]
                rois = preds['rois']
                scores = preds['scores']
                class_ids = preds['class_ids']

                if len(annot) == 1 and len(np.unique(annot[0].cpu().numpy())) == 1:
                    for prediction in rois:
                        false_positive_tmp += 1

                else:
                    for y in annot:
                        if rois.shape[0] == 0:
                            false_negative_tmp += 1
                            continue
                        #print(rois)
                        #print(y)
                        #print(get_max_iou(rois, y.numpy()))
                        max_iou = get_max_iou(rois, y.numpy())
                        #print(max_iou[1])
                        if max_iou[1] > 0.5:
                            true_positive_tmp += 1
                        else:
                            false_negative_tmp += 1
                    if len(annot) < rois.shape[0]:
                        false_positive_tmp += rois.shape[0] - true_positive_tmp

                #print(true_positive_tmp)
                #print(false_negative_tmp)
                #print(false_positive_tmp)
                if str(p) not in precision_recal_dict:
                    precision_recal_dict[str(p)] = [true_positive_tmp, false_positive_tmp,
                                                    false_negative_tmp]
                else:
                    precision_recal_dict[str(p)][0] = precision_recal_dict[str(p)][0] + true_positive_tmp
                    precision_recal_dict[str(p)][1] = precision_recal_dict[str(p)][1] + false_positive_tmp
                    precision_recal_dict[str(p)][2] = precision_recal_dict[str(p)][2] + false_negative_tmp


        miou = np.mean(ious)
        print("mIoU:", miou)
        precisions = []
        recalls = []
        for key in precision_recal_dict:
            value = precision_recal_dict[key]
            if value[0] + value[1] != 0:
                precision = value[0] / (value[0] + value[1])
            else:
                precision = 1
            if value[0] + value[2] !=0:
                recall = value[0] / (value[0] + value[2])
            else:
                recall = 1
            precisions.append(precision)
            recalls.append(recall)
#             print('key: ' + key)
#             print('precision: ' + str(precision))
#             print('recall: ' + str(recall))
#             print('---')
#     print(precisions)
#     print(recalls)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    print('MAP: ', AP)
    plt.plot(precisions, recalls)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()
