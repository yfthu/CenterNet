from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import time
from random import sample
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from utils.visualizer import Visualizer
from progress.bar import Bar
from utils.utils import AverageMeter

def run_epoch(phase, model3d, epoch, data_loader, opt, vis=None):
    # model_with_loss = self.model_with_loss
    # if phase == 'train':
    #     model_with_loss.train()
    # else:
    #     if len(self.opt.gpus) > 1:
    #         model_with_loss = self.model_with_loss.module
    #     model_with_loss.eval()
    #     torch.cuda.empty_cache()
    # todo: 是否需要处理上述代码？

    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    ifshown = sample(range(num_iters), 5) if (phase == 'val' and opt.debug) else []
    for iter_id, batch in enumerate(data_loader):
        if iter_id >= num_iters:
            break
        data_time.update(time.time() - end)

        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        #output, loss, loss_stats = model_with_loss(batch)

        outputs = model3d(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats



        loss = loss.mean()
        if phase == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
            epoch, iter_id, num_iters, phase=phase,
            total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:
            avg_loss_stats[l].update(
                loss_stats[l].mean().item(), batch['input'].size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if not opt.hide_data_time:
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0:
            if iter_id % opt.print_iter == 0:
                print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
        else:
            bar.next()

        if iter_id in ifshown:
            self.debug(batch, output, epoch, iter_id)

        if opt.test:
            self.save_result(output, batch, results)
        del output, loss, loss_stats

    # if epoch % opt.display_freq == 0 and vis is not None:  # display images on visdom and save images to a HTML file
    #   save_result = epoch % opt.update_html_freq == 0
    #   vis.display_current_results(self.get_current_visuals(output, batch, results), epoch, save_result)

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results


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
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
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

    run_epoch("train", epoch, train_loader)

    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    log_dict_train.pop('time', None)
    if opt.display_id > 0:
        vis.plot_current_losses(epoch, float(epoch)/opt.num_epochs, log_dict_train)
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')

    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().init()
  main(opt)