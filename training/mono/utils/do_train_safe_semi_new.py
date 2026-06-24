import os
import torch
import matplotlib.pyplot as plt
from mono.model.monodepth_model import get_configured_monodepth_model
from tensorboardX import SummaryWriter
from mono.utils.comm import TrainingStats
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.seg_metric import IoUMetric
from mono.utils.running import build_lr_schedule_with_cfg, build_optimizer_with_cfg, load_ckpt, save_ckpt, \
    save_best_ckpt, save_best_ckpt_semi
from mono.utils.comm import reduce_dict, main_process, get_rank
from mono.utils.visualization import save_val_imgs, visual_train_data, create_html, save_normal_val_imgs
import traceback
from mono.utils.visualization import create_dir_for_validate_meta, create_dir_for_validate_meta_semi
from mono.model.criterion import build_criterions
from mono.datasets.distributed_sampler import build_dataset_n_sampler_with_cfg, build_data_array, \
    build_dataset_n_sampler_with_cfg_semi
from mono.utils.logger import setup_logger
import logging
from .misc import NativeScalerWithGradNormCount, is_bf16_supported
import math
import sys
import random
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
import tqdm
import copy
from mmengine.model import ExponentialMovingAverage as EMA


def to_cuda(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v) > 1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data


def do_train_safe_semi_new(local_rank: int, cfg: dict):
    logger = setup_logger(cfg.log_file)

    # build criterions
    criterions = build_criterions(cfg)

    # build model
    model = get_configured_monodepth_model(cfg,
                                           criterions,
                                           )
    
    if cfg.get('use_lora', False):
        from peft import LoraConfig, get_peft_model
        
        # 使用正则表达式：匹配模型中名称包含 'backbone' 且以 'qkv' (或其他目标层) 结尾的模块
        # 根据你的 ViT 结构，目标通常是 'qkv' 或者 'query', 'key', 'value'
        # 这里的正则表示：任意字符 + backbone + 任意字符 + (qkv)
        target_regex = r".*encoder.*(?:qkv|query|key|value)"
        
        lora_config = LoraConfig(
            r=cfg.lora_r, 
            lora_alpha=cfg.lora_alpha, 
            target_modules=target_regex  # 传入正则表达式而不是列表
        )
        
        # 直接把最外层的 model 传进去，PEFT 会根据正则自动精准下发 LoRA
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数，你可以借此检查是不是只有 backbone 的层被注入了
        model.print_trainable_parameters()

    # log model state_dict
    if main_process():
        logger.info(model.state_dict().keys())
        for key in model.state_dict().keys():
            if 'seg' in key:
                logger.info(key)

    # build datasets
    train_dataset, train_sampler = build_dataset_n_sampler_with_cfg_semi(cfg, 'train')
    if 'multi_dataset_eval' in cfg.evaluation and cfg.evaluation.multi_dataset_eval:
        val_dataset = build_data_array(cfg, 'val')
    else:
        val_dataset, val_sampler = build_dataset_n_sampler_with_cfg(cfg, 'val')
    # build data loaders
    g = torch.Generator()
    g.manual_seed(cfg.seed + cfg.dist_params.global_rank)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.batchsize_per_gpu,
                                                   num_workers=cfg.thread_per_gpu,
                                                   sampler=train_sampler,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   generator=g, )
    #    collate_fn=collate_fn)
    if isinstance(val_dataset, list):
        val_dataloader = [torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=1,
                                                      num_workers=0,
                                                      sampler=torch.utils.data.distributed.DistributedSampler(
                                                          val_dataset, shuffle=False),
                                                      drop_last=True,
                                                      pin_memory=True, ) for val_group in val_dataset for val_dataset in
                          val_group]
    else:
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=1,
                                                     num_workers=0,
                                                     sampler=val_sampler,
                                                     drop_last=True,
                                                     pin_memory=True, )

    # build schedule
    lr_scheduler = build_lr_schedule_with_cfg(cfg)
    optimizer = build_optimizer_with_cfg(cfg, model)

    # config distributed training
    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    # init automatic mix precision training
    # if 'AMP' in cfg.runner.type:
    #     loss_scaler = NativeScalerWithGradNormCount()
    # else:
    #     loss_scaler = None
    loss_scaler = None

    # load ckpt
    if cfg.load_from and cfg.resume_from is None:
        model, _, _, loss_scaler = load_ckpt(cfg.load_from, model, optimizer=None, scheduler=None, strict_match=False,
                                             loss_scaler=loss_scaler)

        if cfg.freeze_depth:
            for name, param in model.named_parameters():
                if 'safe' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            for name, param in model.named_parameters():
                logger.info(f"{name}: requires_grad={param.requires_grad}")
    elif cfg.resume_from:
        model, optimizer, lr_scheduler, loss_scaler = load_ckpt(
            cfg.resume_from,
            model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            strict_match=False,
            loss_scaler=loss_scaler)

    ########################  加载权重后创建教师模型 ########################
    # teacher_model = copy.deepcopy(model)
    # for param in teacher_model.parameters():
    #     param.requires_grad_(False)
    # teacher_model.eval()

    ##########################################################################

    if cfg.runner.type == 'IterBasedRunner_AMP':
        train_by_iters_amp(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_scaler=loss_scaler
        )
    elif cfg.runner.type == 'EpochBasedRunner':
        raise RuntimeError('It is not supported currently. :)')
    else:
        raise RuntimeError('It is not supported currently. :)')


def split_data_by_label(data_dict):
    """
    根据数据名称中是否包含"Unlabel"拆分数据集

    参数:
    data_dict (dict): 输入的数据字典

    返回:
    tuple: (labeled_dict, unlabeled_dict)
    """
    # 获取 dataset 列表
    dataset = data_dict.get('dataset', [])

    # 创建 unlabel 和 label 的索引
    unlabel_indices = [i for i, name in enumerate(dataset) if "Unlabel" in name]
    label_indices = [i for i, name in enumerate(dataset) if "Unlabel" not in name]

    # 创建结果字典
    labeled_dict = {}
    unlabeled_dict = {}

    # 遍历原始字典的所有项
    for key, value in data_dict.items():
        # print(key)
        # 跳过 dataset 字段
        # if value is None:
        #     continue
        if key == 'dataset':
            labeled_dict['dataset'] = [dataset[i] for i in label_indices]
            unlabeled_dict['dataset'] = [dataset[i] for i in unlabel_indices]
            continue

        # 处理 tensor 类型
        if isinstance(value, torch.Tensor):
            labeled_dict[key] = value[label_indices]
            unlabeled_dict[key] = value[unlabel_indices]

        # 处理 list 类型
        elif isinstance(value, list):
            # 检查列表中是否全是 tensor
            if all(isinstance(item, torch.Tensor) for item in value):
                # 如果全是 tensor，则对每个 tensor 按索引切分
                labeled_list = [item[label_indices] for item in value]
                unlabeled_list = [item[unlabel_indices] for item in value]

                labeled_dict[key] = labeled_list
                unlabeled_dict[key] = unlabeled_list
            elif all(isinstance(item, list) for item in value):
                llist = []
                ulist = []
                for item in value:
                    labeled_list = [item[i] for i in label_indices]
                    unlabeled_list = [item[i] for i in unlabel_indices]
                    llist.append(labeled_list)
                    ulist.append(unlabeled_list)
                labeled_dict[key] = llist
                unlabeled_dict[key] = ulist
            else:
                # 如果不是全是 tensor，直接按索引切分列表
                labeled_dict[key] = [value[i] for i in label_indices]
                unlabeled_dict[key] = [value[i] for i in unlabel_indices]

        # # 对于其他类型，直接复制
        else:
            labeled_dict[key] = value
            unlabeled_dict[key] = value

    return labeled_dict, unlabeled_dict


def cal_kl_loss(teacher_out, student_out, loss_weight=3, loss_gamma=0.9, t=2):
    loss = 0
    n = len(teacher_out)
    for i in range(n):
        adjusted_loss_gamma = loss_gamma ** (15 / (n - 1))
        i_weight = adjusted_loss_gamma ** (n - i - 1)

        student_log_probs = F.log_softmax(student_out[i]/t, dim=1)
        teacher_probs = F.softmax(teacher_out[i]/t, dim=1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='mean')
        loss += i_weight * kl_loss
    return loss_weight * loss


def train_by_iters_amp(cfg, model, optimizer, lr_scheduler, train_dataloader, val_dataloader,
                       loss_scaler):
    """
    Do the training by iterations.
    Mix precision is employed.
    """
    # set up logger
    tb_logger = None
    if cfg.use_tensorboard and main_process():
        tb_logger = SummaryWriter(cfg.tensorboard_dir)
    logger = logging.getLogger()
    # training status
    if main_process():
        training_stats = TrainingStats(log_period=cfg.log_interval, tensorboard_logger=tb_logger)

    # learning schedule
    lr_scheduler.before_run(optimizer)

    # set training steps
    max_iters = cfg.runner.max_iters
    start_iter = lr_scheduler._step_count

    save_interval = cfg.checkpoint_config.interval
    eval_interval = cfg.evaluation.interval
    epoch = 0

    # If it's too slow try lowering num_worker
    # see https://discuss.pytorch.org/t/define-iterator-on-dataloader-is-very-slow/52238
    logger.info('Create iterator.')
    dataloader_iterator = iter(train_dataloader)

    val_err = {}
    seg_val_err = {}
    teacher_val_err = {}
    teacher_seg_val_err = {}
    teacher_model = None
    semi_start_iter = cfg.semi_start_iter
    ema_start_iter = cfg.ema_start_iter
    ema_momentum = cfg.ema_momentum
    # torch.cuda.empty_cache()
    logger.info('Start training.')

    try:
        acc_batch = cfg.acc_batch
    except:
        acc_batch = 1

    try:
        # for step in range(start_iter, max_iters):
        # keep same step in all processes, avoid stuck during eval barrier
        step = start_iter * acc_batch
        cur_iter = 0
        # while step < max_iters:
        while True:

            if main_process():
                training_stats.IterTic()

            # get the data batch
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)
            except Exception as e:
                logger.info('When load training data: ', e)
                continue
            except:
                logger.info('Some training data errors exist in the current iter!')
                continue

            data = to_cuda(data)

            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            #     pred_depth, losses_dict, conf = model(data)
            label_data, unlabel_data = split_data_by_label(data)
            unlabel_data_weak = copy.deepcopy(unlabel_data)
            unlabel_data_weak['input'] = unlabel_data['weak_input']
            del data
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # pred_depth, losses_dict, conf = model(data)
                stu_output = model.module.forward_out(label_data)
            stu_output.update(label_data)
            # labeled_dict, unlabeled_dict = split_data_by_label(stu_output)
            losses_dict = model.module.get_loss(stu_output)

            if cur_iter == ema_start_iter:
                teacher_model = EMA(model, momentum=ema_momentum)
                if cfg.freeze_when_ema:
                    for param in model.module.parameters():
                        param.requires_grad = True
            if cur_iter > semi_start_iter:
                # unlabel_data = {k: unlabeled_dict[k] for k in unlabeled_dict if k in data}
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    teacher_output = teacher_model.module.module.forward_out(unlabel_data_weak)
                    stu_strong_output = model.module.forward_out(unlabel_data)
                stu_strong_output.update(unlabel_data)
                # conf_thresh = 0.8
                depth_pseudo_target = teacher_output['prediction']
                depth_confidence = teacher_output['confidence']
                # depth_pseudo_target[depth_confidence < conf_thresh] = -1
                # depth_confidence = torch.sigmoid(depth_confidence)
                depth_mask = (depth_confidence < cfg.depth_pseudo_thresh[0]) & (
                            depth_confidence > cfg.depth_pseudo_thresh[1])
                depth_pseudo_target = torch.where(depth_mask,
                                                  torch.tensor(-1.0, device=depth_pseudo_target.device),
                                                  depth_pseudo_target)
                stu_strong_output['target'] = depth_pseudo_target

                if cfg.with_safe:
                    safe_teacher_pre = teacher_output['safe_prediction']
                    safe_confidence = torch.softmax(safe_teacher_pre, dim=1)
                    pseudo_mask = safe_confidence > cfg.safe_pseudo_thresh
                    safe_pseudo_target = torch.argmax(safe_confidence, dim=1)
                    pseudo_mask = pseudo_mask.any(dim=1, keepdim=True)
                    pseudo_mask = ~pseudo_mask
                    pseudo_mask = pseudo_mask.squeeze(1)

                    # true_count = torch.sum(pseudo_mask)
                    # false_count = pseudo_mask.numel() - true_count
                    # true_false_ratio = true_count / false_count
                    # safe_pseudo_target = safe_pseudo_target.unsqueeze(1)

                    # safe_pseudo_target[pseudo_mask] = 2.0
                    safe_pseudo_target = torch.where(pseudo_mask,
                                                     torch.tensor(2.0, device=safe_pseudo_target.device),
                                                     safe_pseudo_target)
                    # safe_pseudo_target = safe_pseudo_target.squeeze(1)
                    # b = torch.unique(safe_pseudo_target)
                    stu_strong_output['safe_target'] = safe_pseudo_target
                    if cfg.use_kl:
                        kl_loss = cal_kl_loss(teacher_output['safe_predictions_list'], stu_strong_output['safe_predictions_list'])

                unlabel_loss_dict = model.module.get_loss(stu_strong_output)
                semi_losses_dict = {f"semi_{key}": value*cfg.semi_weight for key, value in unlabel_loss_dict.items()}
                if cfg.semi_entropy_only:
                    # semi_losses_dict = {f"semi_{key}": value for key, value in unlabel_loss_dict.items() if 'DiceLoss' not in key}
                    semi_dice_loss = semi_losses_dict.pop('semi_decode_SafeStandardDiceLoss')
                    semi_losses_dict['semi_total_loss'] = semi_losses_dict['semi_total_loss'] - semi_dice_loss
                losses_dict.update(semi_losses_dict)
                losses_dict['total_loss'] = losses_dict['total_loss'] + losses_dict['semi_total_loss']
                if cfg.use_kl:
                    losses_dict.update({'kl_loss': kl_loss})
                    losses_dict['total_loss'] = losses_dict['total_loss'] + kl_loss

            # if cfg.freeze_depth:
            #     losses_dict = {key: value for key, value in losses_dict.items() if 'Safe' in key}
            #     losses_dict['total_loss'] = torch.sum(torch.stack(list(losses_dict.values())))
            #     print(losses_dict)
                # for key, value in semi_losses_dict.items():
                #     if 'total' not in key:
                #         losses_dict['total_loss'] = losses_dict['total_loss'] + value

            total_loss = losses_dict['total_loss'] / acc_batch

            if not math.isfinite(total_loss):
                logger.info("Loss is {}, skiping this batch training".format(total_loss))
                continue

            # optimize, backward
            if (step + 1 - start_iter) % acc_batch == 0:
                optimizer.zero_grad()
                cur_iter += 1
            if loss_scaler == None:
                total_loss.backward()
                # try:
                if (step + 1 - start_iter) % acc_batch == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
                    optimizer.step()
                    # for param_avg, param_src in zip(teacher_model.parameters(), model.parameters()):
                    #     if param_src.shape != param_avg.shape:
                    #         # param_avg.data = param_src.data.expand_as(param_avg).to(param_avg.device)
                    #         print(param_src.shape, param_avg.shape)
                    if cur_iter > ema_start_iter:
                        teacher_model.update_parameters(model)
                # except Exception as e:
                #     print(e)
                #     print('NAN gradient, skipping optimizer.step() for this round...')
            else:
                loss_scaler(total_loss, optimizer, clip_grad=5, parameters=model.parameters(), update_grad=True)
                if cur_iter > ema_start_iter:
                    teacher_model.update_parameters(model)

            # reduce losses over all GPUs for logging purposes
            if (step + 1 - start_iter) % acc_batch == 0:
                loss_dict_reduced = reduce_dict(losses_dict)
                lr_scheduler.after_train_iter(optimizer)

                if main_process():
                    training_stats.update_iter_stats(loss_dict_reduced)
                    training_stats.IterToc()
                    training_stats.log_iter_stats(step // acc_batch, optimizer, max_iters, val_err)

                # validate the model
                if cfg.evaluation.online_eval and \
                        ((step + acc_batch) // acc_batch) % eval_interval == 0 and \
                        val_dataloader is not None:
                    # if True:
                    if isinstance(val_dataloader, list):
                        # teacher_c =copy.deepcopy(teacher_model.module.module)
                        # val_err, seg_val_err = validate_multiple_dataset(cfg, ((step + acc_batch) // acc_batch), model,
                        #                                                  val_dataloader, tb_logger, prefix='student')
                        if cur_iter > ema_start_iter:
                            original_model_params = copy.deepcopy(model.module.state_dict())
                            val_err, seg_val_err = validate_multiple_dataset(cfg, ((step + acc_batch) // acc_batch),
                                                                             model,
                                                                             val_dataloader, tb_logger,
                                                                             prefix='student')
                            model.module.load_state_dict(teacher_model.module.module.state_dict())
                            teacher_val_err, teacher_seg_val_err = validate_multiple_dataset(cfg, (
                                    (step + acc_batch) // acc_batch), model,
                                                                                             val_dataloader, tb_logger,
                                                                                             prefix='teacher')
                            model.module.load_state_dict(original_model_params)
                        else:
                            val_err, seg_val_err = validate_multiple_dataset(cfg, ((step + acc_batch) // acc_batch),
                                                                             model,
                                                                             val_dataloader, tb_logger,
                                                                             prefix='student')
                    else:
                        val_err = validate(cfg, ((step + acc_batch) // acc_batch), model, val_dataloader, tb_logger)
                    if main_process():
                        training_stats.tb_log_stats(val_err, step)

                # save checkpoint
                if main_process():
                    if (((step + acc_batch) // acc_batch) % save_interval == 0) or (
                            ((step + acc_batch) // acc_batch) == max_iters):
                        # save_ckpt(cfg, model, optimizer, lr_scheduler, ((step+acc_batch)//acc_batch), epoch, loss_scaler=loss_scaler)
                        save_best_ckpt_semi(cfg, model, optimizer, lr_scheduler, ((step + acc_batch) // acc_batch),
                                            epoch,
                                            val_err, seg_val_err, loss_scaler=loss_scaler, prefix='student')
                        if cur_iter > ema_start_iter:
                            save_best_ckpt_semi(cfg, teacher_model, optimizer, lr_scheduler,
                                                ((step + acc_batch) // acc_batch),
                                                epoch,
                                                teacher_val_err, teacher_seg_val_err, loss_scaler=loss_scaler,
                                                prefix='teacher')

            step += 1


    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)


def validate_multiple_dataset(cfg, iter, model, val_dataloaders, tb_logger, prefix):
    val_errs = {}
    seg_val_metrics = {}
    for val_dataloader in val_dataloaders:
        val_err, seg_metrics = validate(cfg, iter, model, val_dataloader, tb_logger, prefix)
        val_errs.update(val_err)
        if cfg.with_safe:
            seg_val_metrics.update(seg_metrics)
    # mean of all dataset
    mean_val_err = {}
    for k, v in val_errs.items():
        metric = 'AllData_eval/' + k.split('/')[-1]
        if metric not in mean_val_err.keys():
            mean_val_err[metric] = 0
        mean_val_err[metric] += v / len(val_dataloaders)
    val_errs.update(mean_val_err)

    if cfg.with_safe:
        mean_seg_err = {}
        for k, v in seg_val_metrics.items():
            metric = 'AllData_eval/' + k.split('/')[-1]
            if metric not in mean_seg_err.keys():
                mean_seg_err[metric] = 0
            mean_seg_err[metric] += v / len(val_dataloaders)
        seg_val_metrics.update(mean_seg_err)
    else:
        seg_val_metrics = None

    return val_errs, seg_val_metrics


def validate(cfg, iter, model, val_dataloader, tb_logger, prefix):
    """
    Validate the model on single dataset
    """
    model.eval()
    dist.barrier()
    logger = logging.getLogger()
    # prepare dir for visualization data
    save_val_meta_data_dir = create_dir_for_validate_meta_semi(cfg.work_dir, iter, prefix)
    # save_html_path = save_val_meta_data_dir + '.html'
    dataset_name = val_dataloader.dataset.data_name

    save_point = max(int(len(val_dataloader) / 5), 1)
    # save_point = 2
    # depth metric meter
    dam = MetricAverageMeter(cfg.evaluation.metrics)
    seg_metric = IoUMetric(iou_metrics=cfg.evaluation.seg_eval_cfg.metrics,
                           class_names=cfg.evaluation.seg_eval_cfg.class_names,
                           ignore_index=cfg.evaluation.seg_eval_cfg.ignore_index)
    # dam_disp = MetricAverageMeter([m for m in cfg.evaluation.metrics if m[:6]!='normal'])
    for i, data in tqdm.tqdm(enumerate(val_dataloader), desc=f"Validating {dataset_name}", total=len(val_dataloader)):
        # if i % 10 == 0:
        #     logger.info(f'Validation step on {dataset_name}: {i}')
        data = to_cuda(data)
        if prefix == 'teacher':
            # output = model.module.module.inference(data)
            output = model.module.inference(data)
        else:
            output = model.module.inference(data)
        pred_depth = output['prediction']
        pred_depth = pred_depth.squeeze()
        gt_depth = data['target'].cuda(non_blocking=True).squeeze()

        pad = data['pad'].squeeze()
        H, W = pred_depth.shape
        pred_depth = pred_depth[pad[0]:H - pad[1], pad[2]:W - pad[3]]
        gt_depth = gt_depth[pad[0]:H - pad[1], pad[2]:W - pad[3]]
        rgb = data['input'][0, :, pad[0]:H - pad[1], pad[2]:W - pad[3]]
        mask = gt_depth > 0
        # pred_depth_resize = cv2.resize(pred_depth.cpu().numpy(), (torch.squeeze(data['B_raw']).shape[1], torch.squeeze(data['B_raw']).shape[0]))
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, cfg.distributed)
        if cfg.with_safe:
            safe_gt = data['safe_target'].squeeze()
            mask = (safe_gt == 0) | (safe_gt == 1)
            safe_gt[~mask] = 2
            safe_pred = output['safe_predictions_list'][-1].squeeze()
            safe_gt = safe_gt[pad[0]:H - pad[1], pad[2]:W - pad[3]]
            safe_pred = safe_pred[:, pad[0]:H - pad[1], pad[2]:W - pad[3]]
            # safe_pred = F.sigmoid(safe_pred)
            # safe_pred = (safe_pred > cfg.evaluation.seg_eval_cfg.thresh).float()
            # target_h = int(safe_gt.shape[1] * data['scale'].item())
            # target_w = int(safe_gt.shape[0] * data['scale'].item())
            # safe_gt = F.interpolate(
            #     safe_gt.unsqueeze(0).unsqueeze(0),  # 增加 batch 维度
            #     size=(target_h, target_w),
            #     mode='bilinear',
            #     align_corners=False
            # ).squeeze(0).squeeze(0)
            # safe_pred = F.interpolate(
            #     safe_pred.unsqueeze(0),
            #     size=(target_h, target_w),
            #     mode='bilinear',
            #     align_corners=False
            # ).squeeze(0)
            # safe_pred = F.softmax(safe_pred, dim=0)
            # safe_pred = torch.argmax(safe_pred, dim=0)
            # seg_metric.process(safe_pred, safe_gt)
            # 【修复1】：纠正 H(shape[0]) 和 W(shape[1]) 的获取
            target_h = int(safe_gt.shape[0] * data['scale'].item())
            target_w = int(safe_gt.shape[1] * data['scale'].item())
            # print((safe_gt.shape, target_h, target_w))
            # target_h = 989
            # target_w = 1320
            
            # 【修复2】：GT 掩码必须转换为 float 进行插值，且必须使用 'nearest'，最后强转回 .long()
            safe_gt = F.interpolate(
                safe_gt.float().unsqueeze(0).unsqueeze(0),  # 增加 batch 和 channel 维度
                size=(target_h, target_w),
                mode='nearest'  # <--- 绝对不能用 bilinear
            ).squeeze(0).squeeze(0).long() # <--- 强转回整型标签

            # 预测值(Logits)可以使用 bilinear
            safe_pred = F.interpolate(
                safe_pred.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # 【优化】：对于 argmax 来说，做不做 softmax 结果完全一样，省略 softmax 节省显存和计算
            safe_pred = torch.argmax(safe_pred, dim=0)
            
            seg_metric.process(safe_pred, safe_gt)

        # save evaluation results
        if i % save_point == 0 and main_process():
            save_val_imgs(iter,
                          pred_depth,
                          gt_depth,
                          rgb,  # data['input'],
                          dataset_name + '_' + data['filename'][0],
                          save_val_meta_data_dir,
                          tb_logger=tb_logger)

        ## surface normal
        if "normal_out_list" in output.keys():
            normal_out_list = output['normal_out_list']
            pred_normal = normal_out_list[-1][:, :3, :, :]  # (B, 3, H, W)
            gt_normal = data['normal'].cuda(non_blocking=True)
            # if pred_normal.shape != gt_normal.shape:
            #     pred_normal = F.interpolate(pred_normal, size=[gt_normal.size(2), gt_normal.size(3)], mode='bilinear', align_corners=True)

            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0]:H - pad[1], pad[2]:W - pad[3]]
            gt_normal = gt_normal[:, :, pad[0]:H - pad[1], pad[2]:W - pad[3]]
            gt_normal_mask = ~torch.all(gt_normal == 0, dim=1, keepdim=True)
            dam.update_normal_metrics_gpu(pred_normal, gt_normal, gt_normal_mask, cfg.distributed)

            # save valiad normal
            if i % save_point == 0 and main_process():
                save_normal_val_imgs(iter,
                                     pred_normal,
                                     gt_normal,
                                     rgb,  # data['input'],
                                     dataset_name + '_normal_' + data['filename'][0],
                                     save_val_meta_data_dir,
                                     tb_logger=tb_logger)

    # create html for visualization
    merged_rgb_pred_gt = os.path.join(save_val_meta_data_dir, '*_merge.jpg')
    name2path = dict(merg=merged_rgb_pred_gt)  # dict(rgbs=rgbs, pred=pred, gt=gt)
    # if main_process():
    #    create_html(name2path, save_path=save_html_path, size=(256*3, 512))

    # get validation error
    eval_error = dam.get_metrics()
    eval_error = {f'{prefix}_{dataset_name}_eval/{k}': v for k, v in eval_error.items()}
    logger.info(eval_error)
    if cfg.with_safe:
        seg_metrics, print_str = seg_metric.compute_metrics(seg_metric.results)
        logger.info(print_str)
        logger.info(seg_metrics)
    else:
        seg_metrics = None

    # eval_disp_error = {f'{dataset_name}_eval/disp_{k}': v for k,v in dam_disp.get_metrics().items()}
    # eval_error.update(eval_disp_error)

    model.train()

    if 'exclude' in cfg.evaluation and dataset_name in cfg.evaluation.exclude:
        return {}
    return eval_error, seg_metrics


def set_random_crop_size_for_iter(dataloader: torch.utils.data.dataloader.DataLoader, iter: int, size_pool=None):
    if size_pool is None:
        size_pool = [
            # [504, 504], [560, 1008], [840, 1512], [1120, 2016],
            [560, 1008], [840, 1512], [1120, 2016],
            # [480, 768], [480, 960], 
            # [480, 992], [480, 1024], 
            # [480, 1120], 
            # [480, 1280], 
            # [480, 1312],
            # [512, 512], [512, 640], 
            # [512, 960], 
            # [512, 992], 
            # [512, 1024], [512, 1120], 
            # [512, 1216], 
            # [512, 1280],
            # [576, 640], [576, 960], 
            # [576, 992], 
            # [576, 1024],
            # [608, 608], [608, 640], 
            # [608, 960], [608, 1024],
        ]
    random.seed(iter)
    sample = random.choice(size_pool)
    # idx = (iter // 10) % len(size_pool)
    # sample = size_pool[size_idx]

    # random.seed(iter)
    # flg = random.random() <= 1.0
    # if flg:
    crop_size = sample
    # else:
    #     crop_size = [sample[1], sample[0]]

    # set crop size for each dataset
    datasets_groups = len(dataloader.dataset.datasets)
    for i in range(datasets_groups):
        for j in range(len(dataloader.dataset.datasets[i].datasets)):
            dataloader.dataset.datasets[i].datasets[j].set_random_crop_size(crop_size)
    return crop_size
