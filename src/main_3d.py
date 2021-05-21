from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from utils.visualizer import Visualizer
from detectors.detector_factory import detector_factory
import CenterNet_3d


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func, gt=False):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.gt = gt
    if self.gt:
      self.num_joints = [4,3,2,0,2]
      self.get_ann_ids_func = dataset.coco.getAnnIds
      self.load_anns_func = dataset.coco.loadAnns

def main(opt):
  if opt.seed is not None:
      torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  centerNet_model = create_model(opt.arch, opt.heads, opt.head_conv)
  CenterNet_optimizer = torch.optim.Adam(centerNet_model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    centerNet_model, optimizer, start_epoch = load_model(
      centerNet_model, opt.load_model, CenterNet_optimizer, opt.resume, opt.lr, opt.lr_step)



  centerNet_3d = CenterNet_3d.CenterNet_3d(centerNet_model, opt)
  optimizer = torch.optim.Adam(centerNet_3d.parameters(), opt.lr)


  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, centerNet_3d, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  vis = Visualizer(opt)
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    #log_dict_train, _ = trainer.train(epoch, train_loader)

    dataset = Dataset(opt, opt.split)
    Detector = detector_factory[opt.task]
    opt.save_infer_dir = os.path.join(opt.save_dir, opt.split)
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process, gt=(opt.debug == 2)),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    log_dict_train, _ = trainer.train(epoch, data_loader) # todo ziji

    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    log_dict_train.pop('time', None)
    if opt.display_id > 0:
        vis.plot_current_losses(epoch, float(epoch)/opt.num_epochs, log_dict_train)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'CenterNet_3d_{}.pth'.format(mark)),
                 epoch, centerNet_3d, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'CenterNet_3d_best.pth'),
                   epoch, centerNet_3d)
    else:
      save_model(os.path.join(opt.save_dir, 'CenterNet_3d_last.pth'),
                 epoch, centerNet_3d, optimizer)
    logger.write('\n')

    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'CenterNet_3d_{}.pth'.format(epoch)),
                 epoch, centerNet_3d, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().init()
  main(opt)