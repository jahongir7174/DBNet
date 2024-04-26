import copy
import math
import random

import cv2
import numpy
import torch
from pyclipper import *
from shapely import Polygon, ops


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[-1]['lr'])
        scheduler.step(epoch + 1, optimizer)

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def load_checkpoint(model, ckpt, prefix='backbone.'):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().state_dict()
    ckpt = {}
    for k, v in src.items():
        if prefix + k in dst and v.shape == dst[prefix + k].shape:
            ckpt[prefix + k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def weight_decay(model):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': 2E-4}]


def mask_to_box(targets, outputs, threshold=0.3, is_polygon=False):
    min_size = 3
    box_threshold = 0.6
    max_candidates = 100

    segmentation = outputs > threshold
    height, width = targets['shape'][0]
    if is_polygon:
        boxes, scores = polygons_from_bitmap(outputs[0],
                                             segmentation[0],
                                             width, height, min_size, max_candidates, box_threshold)
    else:
        boxes, scores = boxes_from_bitmap(outputs[0],
                                          segmentation[0],
                                          width, height, min_size, max_candidates, box_threshold)

    return [boxes], [scores]


def boxes_from_bitmap(output, bitmap, width, height, min_size, max_candidates, box_threshold):
    assert bitmap.size(0) == 1
    bitmap = bitmap.cpu().numpy()[0]
    output = output.cpu().detach().numpy()[0]

    h, w = bitmap.shape

    contours, _ = cv2.findContours((bitmap * 255).astype(numpy.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = min(len(contours), max_candidates)
    boxes = numpy.zeros((num_contours, 4, 2), dtype=numpy.int16)
    scores = numpy.zeros((num_contours,), dtype=numpy.float32)

    for index in range(num_contours):
        contour = contours[index]
        points, small_side = get_mini_boxes(contour)
        if small_side < min_size:
            continue
        points = numpy.array(points)
        score = box_score_fast(output, points.reshape(-1, 2))
        if score < box_threshold:
            continue

        box = un_clip_box(points).reshape(-1, 1, 2)
        box, small_side = get_mini_boxes(box)
        if small_side < min_size + 2:
            continue
        box = numpy.array(box)

        if isinstance(width, torch.Tensor):
            width = width.item()
            height = height.item()

        box[:, 0] = numpy.clip(numpy.round(box[:, 0] / w * width), 0, width)
        box[:, 1] = numpy.clip(numpy.round(box[:, 1] / h * height), 0, height)
        boxes[index, :, :] = box.astype(numpy.int16)
        scores[index] = score
    return boxes, scores


def polygons_from_bitmap(output, bitmap, width, height, min_size, max_candidates, box_threshold):
    assert bitmap.size(0) == 1
    bitmap = bitmap.cpu().numpy()[0]  # The first channel
    output = output.cpu().detach().numpy()[0]
    h, w = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(numpy.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(output, points.reshape(-1, 2))
        if box_threshold > score:
            continue

        if points.shape[0] > 2:
            box = un_clip_mask(points)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, small_side = get_mini_boxes(box.reshape((-1, 1, 2)))
        if small_side < min_size + 2:
            continue

        if isinstance(width, torch.Tensor):
            width = width.item()
            height = height.item()

        box[:, 0] = numpy.clip(numpy.round(box[:, 0] / w * width), 0, width)
        box[:, 1] = numpy.clip(numpy.round(box[:, 1] / h * height), 0, height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


def un_clip(box, ratio):
    polygon = Polygon(box)
    distance = polygon.area * ratio / polygon.length
    offset = PyclipperOffset()
    offset.AddPath(box, JT_ROUND, ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded


def un_clip_box(box):
    return numpy.array(un_clip(box, ratio=1.5))


def un_clip_mask(box):
    expanded = un_clip(box, ratio=2.0)
    if len(expanded) > 1:
        expanded = ops.unary_union([Polygon(poly) for poly in expanded])
        expanded = numpy.array(expanded.exterior.coords)
        return expanded
    else:
        return numpy.array(expanded)


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    x1 = numpy.clip(numpy.floor(box[:, 0].min()).astype(numpy.int32), 0, w - 1)
    x2 = numpy.clip(numpy.ceil(box[:, 0].max()).astype(numpy.int32), 0, w - 1)
    y1 = numpy.clip(numpy.floor(box[:, 1].min()).astype(numpy.int32), 0, h - 1)
    y2 = numpy.clip(numpy.ceil(box[:, 1].max()).astype(numpy.int32), 0, h - 1)

    mask = numpy.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=numpy.uint8)
    box[:, 0] = box[:, 0] - x1
    box[:, 1] = box[:, 1] - y1
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(numpy.int32), 1)
    return cv2.mean(bitmap[y1:y2 + 1, x1:x2 + 1], mask)[0]


def compute_iou(poly1, poly2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        detMatched = 0

        iouMat = numpy.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        for n in range(len(gt)):
            points = gt[n]['points']
            ignore = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if ignore:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if precision > self.area_precision_constraint:
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        if len(gtPols) > 0 and len(detPols) > 0:
            outputShape = [len(gtPols), len(detPols)]
            iouMat = numpy.empty(outputShape)
            gtRectMat = numpy.zeros(len(gtPols), numpy.int8)
            detRectMat = numpy.zeros(len(detPols), numpy.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = compute_iou(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                        detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        perSampleMetrics = {'precision': precision,
                            'recall': recall,
                            'hmean': hmean,
                            'pairs': pairs,
                            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
                            'gtPolPoints': gtPolPoints,
                            'detPolPoints': detPolPoints,
                            'gtCare': numGtCare,
                            'detCare': numDetCare,
                            'gtDontCare': gtDontCarePolsNum,
                            'detDontCare': detDontCarePolsNum,
                            'detMatched': detMatched}

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        precision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        recall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        h_mean = 0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)

        return {'precision': precision, 'recall': recall, 'hmean': h_mean}


class QuadMeasurer:
    def __init__(self, is_polygon):
        self.is_polygon = is_polygon
        self.evaluator = DetectionIoUEvaluator()

    def validate_measure(self, batch, output, box_thresh=0.6):
        results = []
        gt_polygons_batch = batch['polygons']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = numpy.array(output[0], dtype=object)
        pred_scores_batch = numpy.array(output[1], dtype=object)
        for polygons, pred_polygons, pred_scores, ignore_tags in \
                zip(gt_polygons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [dict(points=polygons[i], ignore=ignore_tags[i]) for i in range(len(polygons))]
            if self.is_polygon:
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        pred.append(dict(points=pred_polygons[i, :, :].tolist()))
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results

    def gather_measure(self, raw_metrics):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        f1 = 2 * precision.avg * recall.avg / (precision.avg + recall.avg + 1e-8)

        return precision.avg, recall.avg, f1


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class CosineLR:
    def __init__(self, args, optimizer):
        self.min_lr = 1E-4
        self.epochs = args.epochs
        self.learning_rates = [x['lr'] for x in optimizer.param_groups]

    def step(self, epoch, optimizer):
        param_groups = optimizer.param_groups
        for param_group, lr in zip(param_groups, self.learning_rates):
            alpha = math.cos(math.pi * epoch / self.epochs)
            lr = 0.5 * (lr - self.min_lr) * (1 + alpha)
            param_group['lr'] = self.min_lr + lr


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class DiceLoss(torch.nn.Module):
    """
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity between tow heatmaps.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.k = 50
        self.eps = eps
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, b, t, target, mask):
        """
        output: (N, 1, H, W)
        target: (N, 1, H, W)
        mask: (N, H, W)
        """
        b = self.sigmoid(b)
        t = self.sigmoid(t)
        output = torch.reciprocal(1 + torch.exp(-self.k * (b - t)))
        assert output.dim() == 4, output.dim()
        if output.dim() == 4:
            output = output[:, 0, :, :]
            target = target[:, 0, :, :]

        assert output.shape == target.shape
        assert output.shape == mask.shape

        intersection = (output * target * mask).sum()
        union = (output * mask).sum() + (target * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, output, target, mask):
        output = self.sigmoid(output)
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        else:
            return (torch.abs(output[:, 0] - target) * mask).sum() / mask_sum


class BalancedCELoss(torch.nn.Module):
    """
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.
    """

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target, mask):
        """
        Args:
            output: shape :math:`(N, 1, H, W)`, the prediction of network
            target: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        positive = (target[:, 0, :, :] * mask).byte()
        negative = ((1 - target[:, 0, :, :]) * mask).byte()

        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))

        loss = self.bce(output, target)[:, 0, :, :]

        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        return (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)


class ComputeLoss(torch.nn.Module):
    def __init__(self, eps=1E-6, l1_scale=10, bce_scale=5):
        super().__init__()
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalancedCELoss()
        self.dice_loss = DiceLoss(eps=eps)

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, outputs, targets):
        b = outputs[0]
        t = outputs[1]
        device = outputs[0].device

        b_true = targets['b'].to(device)
        t_true = targets['t'].to(device)
        b_mask = targets['b_mask'].to(device)
        t_mask = targets['t_mask'].to(device)

        bce_loss = self.bce_loss(b, b_true, b_mask)
        dice_loss = self.dice_loss(b, t, b_true, b_mask)
        mask_loss = self.l1_loss(t, t_true, t_mask)
        return bce_loss * self.bce_scale + dice_loss + self.l1_scale * mask_loss
