import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append('/data/tangchen/MDDG')
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from tools.val_mm import evaluate
import wandb
import shutil
from semseg.metrics import Metrics
import gc
import math
from glob import glob


def copyfile(srcfile, dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath,exist_ok=True)                       # 创建路径
        shutil.copy(srcfile, dstpath / fname)          # 复制文件


def main(cfg, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 4
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = cfg['GPUs']
    use_wandb = cfg['USE_WANDB']
    wandb_name = cfg['WANDB_NAME']
    # gpus = int(os.environ['WORLD_SIZE'])
    cls_weights = torch.Tensor(loss_cfg['CLS_WEIGHTS']).to(device) if loss_cfg['CLS_WEIGHTS'] is not None else None
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill = dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = trainset.CLASSES

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'],loss_cfg['AUX_LOSS'])
        
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])
        # print(msg)
        logger.info(msg)
    else:
        model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] 
    
    # loss init #
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    Dice_loss = get_loss(loss_fn_name='Dice',ignore_label = trainset.ignore_label, cls_weights = cls_weights )
    SoftCrossEntropy_loss = get_loss(loss_fn_name='SoftCrossEntropyLoss',ignore_label = trainset.ignore_label, cls_weights = None)
    if loss_cfg['AUX_LOSS']:
        Aux_loss1 = get_loss(loss_fn_name = 'CrossEntropy',ignore_label = trainset.ignore_label, cls_weights = None)
        Aux_loss2 = get_loss(loss_fn_name = 'CrossEntropy',ignore_label = trainset.ignore_label, cls_weights = None)
    # optimize init #
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY']) # 设置优化器
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs + 1)*iters_per_epoch / gpus), sched_cfg['POWER'], int(iters_per_epoch * sched_cfg['WARMUP']/gpus), sched_cfg['WARMUP_RATIO'])

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=False)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None
    
    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss = resume_checkpoint['loss']        
        best_mIoU = resume_checkpoint['best_miou']
        del resume_checkpoint
           
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=sampler_val)

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        model.eval()
        writer = SummaryWriter(str(save_dir))
        # logger.info('================== model complexity =====================')
        # cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)
        logger.info('================== parameter count =====================')
        logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # torch.distributed.barrier()
    for epoch in range(start_epoch, epochs):
        # Clean Memory
        # torch.cuda.empty_cache()
        # gc.collect()

        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0        
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=math.ceil(iters_per_epoch/gpus), desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch / gpus}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        metrics = Metrics(trainset.n_classes, trainloader.dataset.ignore_label, device)

        for iter, (sample, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)
            
            with autocast(enabled=train_cfg['AMP']):
                
                if loss_cfg['AUX_LOSS']:
                    logits, aux_pre1, aux_pre2 = model(sample)
                    loss = 0.5 * Dice_loss(logits, lbl) + 0.5 * SoftCrossEntropy_loss(logits, lbl) + 0.3 * Aux_loss1(aux_pre1, lbl) + 0.3 * Aux_loss2(aux_pre2, lbl)
                    # loss = Dice_loss(logits, lbl)**2 + SoftCrossEntropy_loss(logits, lbl)**2 + Aux_loss1(aux_pre1, lbl)**2 + Aux_loss2(aux_pre2, lbl)**2
                else:
                    logits = model(sample)
                    loss = 0.5 * Dice_loss(logits, lbl) + 0.5 * SoftCrossEntropy_loss(logits, lbl)
                # loss = loss_fn(logits, lbl)

            metrics.update(logits.softmax(dim=1).detach(), lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8 # minimum of lr
            train_loss += loss.item()
            
            # Clean Memory
            # torch.cuda.empty_cache()
            gc.collect()
            scheduler.step()
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)

        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        # if use_wandb:
        train_log_data = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train mIoU": miou,
            "Train Pixel Acc": macc,
            "Train F1": mf1,
        }

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            #要改成四卡并发
            
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, f1, mf1, ious, miou, test_loss = evaluate(model, valloader, device, loss_fn = loss_fn)
                writer.add_scalar('val/mIoU', miou, epoch)

                # if use wandb
                log_data = {
                    "Test Loss": test_loss,
                    "Test mIoU": miou,
                    "Test Pixel Acc": macc,
                    "Test F1": mf1,
                }
                log_data.update(train_log_data)
                print(log_data)
                if use_wandb:
                    wandb.log(log_data)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch + 1
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    # torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    torch.save(model.module.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                else:
                    last_pth = save_dir / f"last_checkpoint.pth"

                logger.info(f"Current epoch:{epoch} Lr:{lr} Train Loss:{train_loss} Test mIoU: {miou} Test Best mIoU: {best_mIoU}")
        
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/mcubes_rgbadn.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    setup_cudnn()
    gpu = setup_ddp()
    fix_seeds(3407)
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    # exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    exp_name = cfg['WANDB_NAME']
    if cfg['USE_WANDB']:
        wandb.init(project="ProjcetName", entity="EntityName", name=exp_name)

    save_dir = Path(cfg['SAVE_DIR'], exp_name , time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime()))
    # cfpath,cfname = os.path.split(args.cfg)
    if torch.distributed.get_rank() == 0:
        print("sava config")
        copyfile(args.cfg, save_dir) # 保存配置文件
        # src_file_list = os.listdir('semseg/models')
        shutil.copytree('semseg/models', save_dir, dirs_exist_ok = True)
        # print(src_file_list)# glob获得路径下所有文件，可根据需要修改
        # for srcfile in src_file_list:
        #     copyfile(srcfile, save_dir) 
        # shutil.copy(args.cfg, Path(save_dir, cfname))
                 
    if os.path.isfile(cfg['MODEL']['RESUME']):
            save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))    
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, save_dir)
    cleanup_ddp()