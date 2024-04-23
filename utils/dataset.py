import math
import random

import cv2
import numpy
import torch
from pyclipper import *
from shapely import Polygon
from torch.utils import data


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def resize(image, input_size):
    shape = image.shape
    width = shape[1] * input_size / shape[0]
    width = math.ceil(width / 32) * 32
    return cv2.resize(image, dsize=(width, input_size))


class RandomAffine:
    def __init__(self):
        self.params = {'degrees': 15,
                       'scale': 0.50,
                       'shear': 0.001,
                       'translate': 0.25}

    def __call__(self, image, b, b_mask, t, t_mask):
        h, w = image.shape[:2]

        # Center
        center = numpy.eye(3)
        center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        perspective = numpy.eye(3)

        # Rotation and Scale
        rotate = numpy.eye(3)
        a = random.uniform(-self.params['degrees'], self.params['degrees'])
        s = random.uniform(1 - self.params['scale'], 1 + self.params['scale'])
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        shear = numpy.eye(3)
        shear[0, 1] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)

        # Translation
        translate = numpy.eye(3)
        translate[0, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * w
        translate[1, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * h

        # Combined rotation matrix, order of operations (right to left) is IMPORTANT
        matrix = translate @ shear @ rotate @ perspective @ center
        if numpy.any((matrix != numpy.eye(3))):  # image changed
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
            b = cv2.warpAffine(b, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
            t = cv2.warpAffine(t, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
            b_mask = cv2.warpAffine(b_mask, matrix[:2], dsize=(w, h), borderValue=(1, 1, 1))
            t_mask = cv2.warpAffine(t_mask, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
        return image, b, b_mask, t, t_mask


class RandomHSV:
    def __init__(self,
                 h_delta: int = 5,
                 s_delta: int = 30,
                 v_delta: int = 30):
        self.h_delta = h_delta
        self.s_delta = s_delta
        self.v_delta = v_delta

    def random_hsv(self):
        h = self.h_delta
        s = self.s_delta
        v = self.v_delta
        hsv_gains = numpy.random.uniform(-1, 1, 3) * [h, s, v]
        # random selection of h, s, v
        hsv_gains *= numpy.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(numpy.int16)
        return hsv_gains

    def __call__(self, img):
        hsv = self.random_hsv()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(numpy.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv[0]) % 180
        img_hsv[..., 1] = numpy.clip(img_hsv[..., 1] + hsv[1], 0, 255)
        img_hsv[..., 2] = numpy.clip(img_hsv[..., 2] + hsv[2], 0, 255)
        return cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)


class RandomResize:
    def __init__(self, size):
        self.size = size
        self.scale = (size, size)
        self.ratio_range = (0.1, 2.0)

    def resize(self, x, b, b_mask, t, t_mask):
        min_ratio, max_ratio = self.ratio_range
        assert min_ratio <= max_ratio
        ratio = numpy.random.random_sample()
        ratio = ratio * (max_ratio - min_ratio) + min_ratio
        scale = int(self.scale[0] * ratio), int(self.scale[1] * ratio)

        shape = x.shape[:2]
        scale = min(max(scale) / max(shape), min(scale) / min(shape))

        w = int(shape[1] * scale + 0.5)
        h = int(shape[0] * scale + 0.5)

        x = cv2.resize(x, dsize=(w, h), interpolation=resample())
        b = cv2.resize(b, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        t = cv2.resize(t, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        b_mask = cv2.resize(b_mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        t_mask = cv2.resize(t_mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        return x, b, b_mask, t, t_mask

    def pad(self, x, b, b_mask, t, t_mask):
        shape = (x.shape[1], x.shape[0])
        ratio = float(self.size) / max(shape)
        size = tuple([int(x * ratio) for x in shape])

        if size[0] > self.size or size[1] > self.size:
            ratio = float(self.size) / min(shape)
            size = tuple([int(x * ratio) for x in shape])

        w = self.size - size[0]
        h = self.size - size[1]
        top, bottom = h // 2, h - (h // 2)
        left, right = w // 2, w - (w // 2)
        x = cv2.resize(x, size, interpolation=resample())
        b = cv2.resize(b, size, interpolation=cv2.INTER_NEAREST)
        t = cv2.resize(t, size, interpolation=cv2.INTER_NEAREST)
        b_mask = cv2.resize(b_mask, size, interpolation=cv2.INTER_NEAREST)
        t_mask = cv2.resize(t_mask, size, interpolation=cv2.INTER_NEAREST)
        x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        b = cv2.copyMakeBorder(b, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        t = cv2.copyMakeBorder(t, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        b_mask = cv2.copyMakeBorder(b_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(1, 1, 1))
        t_mask = cv2.copyMakeBorder(t_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return x, b, b_mask, t, t_mask

    def __call__(self, x, b, b_mask, t, t_mask):
        x, b, b_mask, t, t_mask = self.resize(x, b, b_mask, t, t_mask)
        x, b, b_mask, t, t_mask = self.pad(x, b, b_mask, t, t_mask)
        return x, b, b_mask, t, t_mask


class Transform:
    def __init__(self):
        self.shrink_ratio = .4
        self.min_text_size = 8
        self.t_min = 0.3
        self.t_max = 0.7

    def __call__(self, image, polygons, ignore_tags):
        h, w = image.shape[:2]

        b = numpy.zeros(shape=(h, w), dtype=numpy.float32)
        b_mask = numpy.ones(shape=(h, w), dtype=numpy.float32)

        polygons, ignore_tags = self.validate_polygons(polygons, ignore_tags, h, w)
        for i, polygon in enumerate(polygons):
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(b_mask, polygon.astype(numpy.int32)[numpy.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * (1 - numpy.power(self.shrink_ratio, 2)) / polygon_shape.length
                padding = PyclipperOffset()
                padding.AddPath([tuple(p) for p in polygons[i]], JT_ROUND, ET_CLOSEDPOLYGON)
                padded = padding.Execute(-distance)
                if not padded:
                    cv2.fillPoly(b_mask, polygon.astype(numpy.int32)[numpy.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                padded = numpy.array(padded[0]).reshape(-1, 2)
                cv2.fillPoly(b, [padded.astype(numpy.int32)], 1)

        t = numpy.zeros(image.shape[:2], dtype=numpy.float32)
        t_mask = numpy.zeros(image.shape[:2], dtype=numpy.float32)
        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], t, mask=t_mask)
        t = t * (self.t_max - self.t_min) + self.t_min
        return image, b, b_mask, t, t_mask

    def validate_polygons(self, polygons, ignore_tags, h, w):
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = numpy.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = numpy.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    @staticmethod
    def polygon_area(polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

    def draw_border_map(self, polygon, canvas, mask):
        polygon = numpy.array(polygon)

        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - numpy.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(p) for p in polygon]
        padding = PyclipperOffset()
        padding.AddPath(subject, JT_ROUND, ET_CLOSEDPOLYGON)
        padded_polygon = numpy.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(numpy.int32)], 1.0)

        x1 = padded_polygon[:, 0].min()
        x2 = padded_polygon[:, 0].max()
        y1 = padded_polygon[:, 1].min()
        y2 = padded_polygon[:, 1].max()

        w = x2 - x1 + 1
        h = y2 - y1 + 1

        polygon[:, 0] = polygon[:, 0] - x1
        polygon[:, 1] = polygon[:, 1] - y1

        xs = numpy.broadcast_to(numpy.linspace(0, w - 1, num=w).reshape(1, w), (h, w))
        ys = numpy.broadcast_to(numpy.linspace(0, h - 1, num=h).reshape(h, 1), (h, w))

        distance_map = numpy.zeros((polygon.shape[0], h, w), dtype=numpy.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = numpy.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        x1_valid = min(max(0, x1), canvas.shape[1] - 1)
        x2_valid = min(max(0, x2), canvas.shape[1] - 1)
        y1_valid = min(max(0, y1), canvas.shape[0] - 1)
        y2_valid = min(max(0, y2), canvas.shape[0] - 1)

        canvas_valid = canvas[y1_valid:y2_valid + 1, x1_valid:x2_valid + 1]
        distance_map_valid = distance_map[y1_valid - y1:y2_valid - y2 + h, x1_valid - x1:x2_valid - x2 + w]

        canvas[y1_valid:y2_valid + 1, x1_valid:x2_valid + 1] = numpy.fmax(1 - distance_map_valid, canvas_valid)

    @staticmethod
    def distance(xs, ys, point_1, point_2):
        square_distance_1 = numpy.square(xs - point_1[0]) + numpy.square(ys - point_1[1])
        square_distance_2 = numpy.square(xs - point_2[0]) + numpy.square(ys - point_2[1])
        square_distance = numpy.square(point_1[0] - point_2[0]) + numpy.square(point_1[1] - point_2[1])

        cosine = (square_distance - square_distance_1 - square_distance_2) / \
                 (2 * numpy.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - numpy.square(cosine)
        square_sin = numpy.nan_to_num(square_sin)

        result = numpy.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
        result[cosine < 0] = numpy.sqrt(numpy.fmin(square_distance_1, square_distance_2))[cosine < 0]
        return result


class Dataset(data.Dataset):
    def __init__(self, args, filenames, train):
        labels = self.load_labels(filenames)
        self.args = args
        self.train = train
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())

        self.transform = Transform()
        self.random_hsv = RandomHSV()
        self.random_affine = RandomAffine()
        self.random_resize = RandomResize(args.input_size)

        self.mean = numpy.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
        self.std = numpy.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')

    def __getitem__(self, index):
        image = cv2.imread(self.filenames[index], cv2.IMREAD_COLOR).astype('float32')
        label = self.labels[index]
        shape = image.shape

        annotations = []
        for line in label:
            annotations.append({'points': [(p[0], p[1]) for p in line['poly']],
                                'ignore': line['text'] == '###',
                                'text': line['text']})
        polygons = []
        ignore_tags = []

        for annotation in annotations:
            polygons.append(numpy.array(annotation['points']))
            ignore_tags.append(annotation['ignore'])
        ignore_tags = numpy.array(ignore_tags, dtype=numpy.uint8)

        if self.train:
            image, b, b_mask, t, t_mask = self.transform(image, polygons, ignore_tags)
            image, b, b_mask, t, t_mask = self.random_resize(image, b, b_mask, t, t_mask)
            image, b, b_mask, t, t_mask = self.random_affine(image, b, b_mask, t, t_mask)
            image, b, b_mask, t, t_mask = self.flip_lr(image, b, b_mask, t, t_mask)

            image = self.random_hsv(image)
            image = self.to_tensor(image)

            return image, {'b': b[numpy.newaxis, :, :], 'b_mask': b_mask, 't': t, 't_mask': t_mask}
        else:
            image = resize(image, self.args.input_size)
            image = self.to_tensor(image)

            return image, tuple(shape[:2]), polygons, ignore_tags

    def __len__(self):
        return len(self.filenames)

    def to_tensor(self, image):
        image = image.astype('float32') / 255.0
        image = image - self.mean
        image = image / self.std

        image = image.transpose((2, 0, 1))[::-1]
        image = numpy.ascontiguousarray(image)
        image = torch.from_numpy(image)
        return image

    @staticmethod
    def flip_lr(image, b, b_mask, t, t_mask):
        # Flip left-right
        if random.random() < 0.5:
            image = cv2.flip(image, flipCode=1)
            b = cv2.flip(b, flipCode=1)
            t = cv2.flip(t, flipCode=1)
            b_mask = cv2.flip(b_mask, flipCode=1)
            t_mask = cv2.flip(t_mask, flipCode=1)
        return image, b, b_mask, t, t_mask

    @staticmethod
    def load_labels(filenames):
        labels = {}
        for filename in filenames:
            lines = []
            with open(filename.replace('images', 'labels') + '.txt', 'r') as f:
                for line in f.readlines():
                    item = {}
                    parts = line.strip().split(',')
                    line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = numpy.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                    item['poly'] = poly
                    item['text'] = parts[-1]
                    lines.append(item)
            labels[filename] = lines
        return labels

    @staticmethod
    def collate_fn(batch):
        image, shape, polygons, ignore_tags = zip(*batch)
        samples = torch.stack(tensors=image, dim=0)
        targets = {'shape': shape,
                   'polygons': polygons,
                   'ignore_tags': ignore_tags}
        return samples, targets
