import argparse
import copy
import csv
import math
import os
import warnings

import cv2
import numpy
import torch
import tqdm
from timm import utils
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = '../Dataset/TotalText'


def lr(args):
    return 1E-2 * args.batch_size * args.world_size / 16


def train(args):
    # Model
    model = nn.DBNet()
    model = util.load_checkpoint(model, ckpt='./weights/imagenet.pt')
    model.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(util.weight_decay(model), lr(args), momentum=0.9, nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    sampler = None
    filenames = []
    with open('../Dataset/TotalText/train.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip()
            filenames.append('../Dataset/TotalText/images/train/' + filename)

    dataset = Dataset(args, filenames, train=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None,
                             sampler=sampler, num_workers=8, pin_memory=True)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    num_steps = len(loader)

    criterion = util.ComputeLoss().cuda()
    amp_scale = torch.cuda.amp.GradScaler()
    scheduler = util.CosineLR(args, optimizer)
    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'loss', 'Recall', 'Precision', 'F1'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()

            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = loader
            avg_loss = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_steps)

            optimizer.zero_grad()
            for samples, targets in p_bar:
                samples = samples.cuda()

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                amp_scale.step(optimizer)
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

                # Log
                if args.distributed:
                    loss = utils.reduce_tensor(loss.data, args.world_size)
                avg_loss.update(loss.item(), samples.size(0))
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, avg_loss.avg)
                    p_bar.set_description(s)

            # Scheduler
            scheduler.step(epoch, optimizer)

            if args.local_rank == 0:
                last = test(args, ema.ema)

                logger.writerow({'Precision': str(f'{last[0]:.3f}'),
                                 'Recall': str(f'{last[1]:.3f}'),
                                 'F1': str(f'{last[2]:.3f}'),
                                 'loss': str(f'{avg_loss.avg:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                log.flush()

                # Update best F1
                if best < last[2]:
                    best = last[2]

                # Save model
                save = copy.deepcopy(ema.ema)
                save = {'epoch': epoch,
                        'model': save.half()}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[2]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    filenames = []
    with open('../Dataset/TotalText/test.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip()
            filenames.append('../Dataset/TotalText/images/test/' + filename)

    dataset = Dataset(args, filenames, train=False)
    loader = data.DataLoader(dataset, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load(f'./weights/last.pt')
        model = model['model']
        model = model.float()
        model.cuda()

    model.eval()

    results = []

    evaluator = util.QuadMeasurer(is_polygon=True)
    for sample, target in tqdm.tqdm(loader, ('%10s' * 3) % ('precision', 'recall', 'F1')):
        output = model(sample.cuda())
        output = util.mask_to_box(target, output.cpu(), is_polygon=True)

        result = evaluator.validate_measure(target, output)
        results.append(result)
    precision, recall, f1 = evaluator.gather_measure(results)
    # Print results
    print(('%10s' * 3) % (f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}'))

    # Return results
    model.float()  # for training
    return precision, recall, f1


@torch.no_grad()
def demo(args, model=None):
    filenames = []
    with open('../Dataset/TotalText/test.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip()
            filenames.append('../Dataset/TotalText/images/test/' + filename)

    if model is None:
        model = torch.load(f'./weights/last.pt')
        model = model['model']
        model = model.float()
        model.cuda()

    model.eval()
    mean = numpy.array([0.406, 0.456, 0.485]).reshape((1, 1, 3)).astype('float32')
    std = numpy.array([0.225, 0.224, 0.229]).reshape((1, 1, 3)).astype('float32')
    for filename in tqdm.tqdm(filenames):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        shape = image.shape[:2]
        width = shape[1] * args.input_size / shape[0]
        width = math.ceil(width / 32) * 32

        x = cv2.resize(image, dsize=(width, args.input_size))
        x = x.astype('float32') / 255.0
        x = x - mean
        x = x / std
        x = x.transpose((2, 0, 1))[::-1]
        x = numpy.ascontiguousarray(x)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.cuda()

        output = model(x)
        output = util.mask_to_box(targets={'shape': [shape]}, outputs=output.cpu(), is_polygon=True)
        boxes, scores = output[0][0], output[1][0]
        for box in boxes:
            box = numpy.array(box).reshape((-1, 1, 2))
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=5)

        cv2.imwrite(f'./data/{os.path.basename(filename)}', image)


def profile(args):
    import thop
    model = nn.DBNet().fuse()
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(copy.copy(model), inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=800, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)

    image = cv2.imread('./demo/demo.jpg', cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    x = cv2.resize(image, dsize=(w // 4, h // 4))
    cv2.imwrite('./demo/demo.jpg', x)


if __name__ == "__main__":
    main()
