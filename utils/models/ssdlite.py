import torch
import math
import torch.nn.functional as F
from numpy import mgrid
from functools import partial
from torch import nn
from math import floor
from torch.cuda.amp import autocast
from utils.models.ops import get_output_shapes, xavier_init
from utils.models.layers import ConvBNReLU
from utils.constants import BACKGROUND_INDEX, COLOR
from utils.boxes import cxcywh2xyxy, xyxy2cxcywh, calculate_ious
import cv2

class _SSDLiteHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBNReLU(in_channels,
                       in_channels,
                       kernel_size=3,
                       depthwise=True,
                       relu6=True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1),
        )

class _Heads(nn.Module):
    def __init__(self, layer_channels, num_anchor_shapes, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.classifincation_heads = nn.ModuleList([])
        self.regression_heads = nn.ModuleList([])
        for cin, n in zip(layer_channels, num_anchor_shapes):
            self.classifincation_heads.append(_SSDLiteHead(cin, (num_classes + 1) * n))
            self.regression_heads.append(_SSDLiteHead(cin, 4 * n))

        self.apply(xavier_init)

    def forward(self, features):
        regression_preds, class_preds = [], []
        bs = features[0].shape[0]
        for i, ftrs in enumerate(features):
            regression_preds.append(
                self.regression_heads[i](ftrs).reshape([bs, 4, -1])
            )
            class_preds.append(
                self.classifincation_heads[i](ftrs).reshape([bs, (self.num_classes + 1), -1])
            )
        regression_preds = torch.cat(regression_preds, axis=-1)
        class_preds = torch.cat(class_preds, axis=-1)
        return regression_preds, class_preds

class SSDLite(nn.Sequential): 
    def __init__(self, backbone, num_classes, input_size, anchor_scales, anchor_aspect_ratios, device):
        feature_shapes = get_output_shapes(backbone, input_size)   # [num_stages, [3]]
        heads = _Heads(
            layer_channels=[shape[0] for shape in feature_shapes], 
            num_anchor_shapes=[len(ar) * 2 for ar in anchor_aspect_ratios], 
            num_classes=num_classes
        )
        super().__init__(backbone, heads)

        self.device = device

        anchors = self._define_anchors(# pixel coordinates; cxcywh; [num_anchors, 4]
            input_size,
            feature_sizes=[shape[1] for shape in feature_shapes],
            scales=anchor_scales,
            aspect_ratios=anchor_aspect_ratios
        )
        self.register_buffer('anchors', anchors)

    @staticmethod
    def _define_anchors(input_size, feature_sizes, scales, aspect_ratios):
        num_stages = len(feature_sizes)
        anchors = []
        for i in range(num_stages):
            wh = []
            for ar in aspect_ratios[i]:
                if ar == 1:
                    s1 = scales[i]
                    try:
                        s2 = math.sqrt(scales[i] * scales[i + 1])
                    except IndexError:
                        s2 = 1.
                    wh.extend([[s1, s1], [s2, s2]])
                else:
                    w = scales[i] * math.sqrt(ar)
                    h = scales[i] / math.sqrt(ar)
                    wh.extend([[w, h], [h, w]])

            size = feature_sizes[i]
            wh = torch.FloatTensor(wh).T    # [2, n]
            wh = torch.tile(                # [2, n, size, size]
                torch.reshape(wh, [2, -1, 1, 1]),
                [1, 1, size, size]
            )

            cycx = (torch.FloatTensor(mgrid[:size, :size]) + 0.5) / size   # [2, size, size]
            cxcy = cycx[[1, 0]]                                            # [2, size, size]
            cxcy = torch.broadcast_to(cxcy.unsqueeze(1), wh.shape)         # [2, n, size, size]

            layer_anchors = torch.cat([cxcy, wh], axis=0)                  # [4, n, size, size]
            layer_anchors = torch.reshape(layer_anchors, [4, -1])          # [4, n * size * size]
            anchors.append(layer_anchors)
        anchors = torch.cat(anchors, axis=-1)   # [4, num_anchors]
        anchors = anchors.T                     # [num_anchors, 4]
        anchors *= input_size
        return anchors

    def compute_loss(self, preds, true_boxes, true_classes, neg_pos_ratio=3):
        positive_mask, regression_target, classification_target = (
            self._encode_ground_truth(true_boxes, true_classes)
        )

        regression_preds, class_preds = preds

        regression_loss = F.smooth_l1_loss(      # [bs, num_anchors, 4]
            regression_preds.transpose(1, 2),    # [bs, num_anchors, 4]
            regression_target,
            reduction='none'
        )[positive_mask].sum()

        classification_loss = F.cross_entropy(   # [bs, num_anchors]
            class_preds,                         # [bs, num_classes + 1, num_anchors]
            classification_target,               # [bs, num_anchors]
            reduction='none'
        )

        positvie_classification_loss = classification_loss[positive_mask].sum()

        num_positives = torch.sum(positive_mask.int(), axis=-1)   # [bs]
        num_negatives = torch.clip(num_positives, min=1) * neg_pos_ratio
        negative_mask = (   # [bs, num_anchors]
            torch.arange(classification_loss.shape[1])
            < num_negatives.to(self.device).unsqueeze(-1)   # [bs, 1]
        )
        negative_classification_loss, _ = torch.sort(
            classification_loss * (~positive_mask).int(),
            descending=True
        )
        negative_classification_loss = negative_classification_loss[negative_mask].sum()

        total_loss = (
            regression_loss
            + positvie_classification_loss
            + negative_classification_loss
        ) / num_positives.sum()
        return total_loss

    def _encode_ground_truth(self, true_boxes, true_classes):
        xyxy_anchors = cxcywh2xyxy(self.anchors)

        bs = len(true_boxes)
        positive_mask, regression_target, classification_target = [], [], []
        for i in range(bs):
            num_true = true_boxes[i].shape[0]
            ious = calculate_ious(true_boxes[i], xyxy_anchors)   # [num_true, num_anchors]

            # Assign each ground truth box to the anchor with the largest IoU.
            best_anchor_for_ground_truth = torch.argmax(ious, axis=-1)   # [num_true]
            ious[torch.arange(num_true), best_anchor_for_ground_truth] = 1.

            # Ignore backgound
            ious[true_classes[i] == BACKGROUND_INDEX] = 0.

            ground_truth_for_anchor = torch.argmax(ious, axis=0)           # [num_anchors]
            regression_target_i = true_boxes[i][ground_truth_for_anchor]   # [num_anchors, 4]
            regression_target_i = xyxy2cxcywh(regression_target_i)
            regression_target_i = torch.cat(
                [
                    ((regression_target_i[..., :2] - self.anchors[:, :2])
                     / (self.anchors[:, 2:] / 10)),
                    torch.log(regression_target_i[:, 2:] / self.anchors[:, 2:]) * 5,
                ],
                axis=-1
            )
            classification_target_i = 1 + true_classes[i][ground_truth_for_anchor]  # [num_anchors]

            # Determine positive anchors
            best_ious, _ = torch.max(ious, axis=0)   # [num_anchors]
            positive_mask_i = (best_ious > 0.5)
            regression_target_i[~positive_mask_i] = 0.
            classification_target_i[~positive_mask_i] = 0

            positive_mask.append(positive_mask_i)
            regression_target.append(regression_target_i)
            classification_target.append(classification_target_i)
        positive_mask = torch.stack(positive_mask, axis=0)
        regression_target = torch.stack(regression_target, axis=0)
        classification_target = torch.stack(classification_target, axis=0)
        return positive_mask, regression_target, classification_target

    def decode(self, preds):
        regression_preds, class_preds = preds
        bs, _, num_anchors = regression_preds.shape
        num_classes = class_preds.shape[1] - 1

        regression_preds = regression_preds.transpose(1, 2)
        class_preds = class_preds.transpose(1, 2)

        cxcy = (regression_preds[..., :2] * self.anchors[:, 2:] / 10
                + self.anchors[:, :2])                 # [bs, num_anchors, 2]
        wh = (torch.exp(regression_preds[..., 2:] / 5)
              * self.anchors[:, 2:])                   # [bs, num_anchors, 2]
        boxes = torch.cat([cxcy, wh], axis=-1)         # [bs, num_anchors, 4]
        boxes = cxcywh2xyxy(boxes)
        boxes = torch.tile(    # [bs, num_anchors, num_classes, 4]
            torch.unsqueeze(boxes, axis=2),
            [1, 1, num_classes, 1]
        )
        scores = F.softmax(class_preds, dim=-1)[..., 1:]   # [bs, num_anchors, num_classes]
        classes = torch.broadcast_to(
            torch.arange(num_classes, dtype=torch.int64),
            [bs, num_anchors, num_classes]
        )
        classes = classes.to(boxes.device)

        boxes = torch.reshape(boxes, [bs, num_anchors * num_classes, 4])
        scores = torch.reshape(scores, [bs, num_anchors * num_classes])
        classes = torch.reshape(classes, [bs, num_anchors * num_classes])
        return boxes, scores, classes
    
    def detect(self, image_name, image, label_names, logging, threshold=0.5, no_amp=True):
        from utils.misc import nms
        nb_found = 0
        with torch.no_grad():
            with autocast(enabled=(not no_amp)):
                preds = self(image)
        
        det_boxes, det_scores, det_classes = nms(*self.decode(preds))
        image = cv2.cvtColor(image[0].permute(1, 2, 0).to(self.device).numpy(), cv2.COLOR_RGB2BGR)
    
        boxes = []
        labels = []
        scores = []
        detection = ''
        image_info = f'IMAGE : {image_name.split("/")[-1]}'
        
        for box, score, cls in zip(det_boxes[0], det_scores[0], det_classes[0]):
            if score > threshold:
                x1, y1, x2, y2 = box.to(self.device).numpy().astype(int)
                score_value = (str(floor(float(score.to(self.device).numpy()) * 100))) + '%'
                label =  label_names[cls.to(self.device).numpy()] + '-' + score_value
                cv2.rectangle(image, (x1, y1), (x2, y2), COLOR, 1)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, -1)
                image = cv2.rectangle(image, (x1, y1 - 15), (x1 + w, y1), COLOR, -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                nb_found += 1
                detection += f'{nb_found} => {label}\n'
                boxes.append(box)
                scores.append(score)
                labels.append(label_names[cls.to(self.device).numpy()])
                
        if nb_found > 0:
            INDEX = -1
            SEPARATOR = '------------'

            logging.info(image_info)
            logging.info(detection[:INDEX])
            logging.info(SEPARATOR)
            
            print(image_info) 
            print(detection[:INDEX])           
            print(SEPARATOR)
             
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), nb_found, {'image' : image, 'boxes' : boxes, 'labels' : labels, 'scores' : scores}