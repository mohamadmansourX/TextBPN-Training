from numpy.lib.arraypad import pad
import cv2
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from util.misc import AverageMeter
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.textnet import TextNet
from util.augmentation import BaseTransform,Augmentation
from cfglib.config import init_config, update_config, print_config
from cfglib.option import BaseOptions
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs,rescale_result
from util.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, \
    deal_eval_TD500, data_transfer_ICDAR, data_transfer_TD500, data_transfer_MLT2017
import sys
from network.loss import  TextLoss

sys.argv=['']


def main():
    cfg = init_config()
    option = BaseOptions()
    args = option.initialize()
    update_config(cfg, args)
    print_config(cfg)
    
    # Create checkpoint directory
    if not os.path.exists(cfg.save_path):
        mkdirs(cfg.save_path)
    
    # Create the model
    model = TextNet(is_training=True, backbone=cfg.net,)
    model.train()
    # Initialize wandb
    if cfg.wandb_flag:
        global wandb
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg)
        wandb.watch(model)

    # Load the dataset
    if cfg.dataset == 'TotalText':
        train_dataset = TotalText(cfg.train_data_root, is_training=True,
                                transform=Augmentation(cfg.train_input_size, cfg.rgb_mean, cfg.rgb_std))
        val_dataset = TotalText(cfg.val_data_root, is_training=False,
                                transform=BaseTransform(cfg.val_input_size, cfg.rgb_mean, cfg.rgb_std))
    elif cfg.dataset == 'CTW1500':
        train_dataset = Ctw1500Text(cfg.train_data_root, is_training=True,
                                    transform=Augmentation(cfg.train_input_size, cfg.rgb_mean, cfg.rgb_std))
        val_dataset = Ctw1500Text(cfg.val_data_root, is_training=False,
                                transform=BaseTransform(cfg.val_input_size, cfg.rgb_mean, cfg.rgb_std))
    elif cfg.dataset == 'ICDAR15':
        train_dataset = Icdar15Text(cfg.train_data_root, is_training=True,
                                    transform=Augmentation(cfg.train_input_size, cfg.rgb_mean, cfg.rgb_std))
        val_dataset = Icdar15Text(cfg.val_data_root, is_training=False,
                                transform=BaseTransform(cfg.val_input_size, cfg.rgb_mean, cfg.rgb_std))
    elif cfg.dataset == 'MLT2017':
        train_dataset = Mlt2017Text(cfg.train_data_root, is_training=True,
                                    transform=Augmentation(cfg.train_input_size, cfg.rgb_mean, cfg.rgb_std))
        val_dataset = Mlt2017Text(cfg.val_data_root, is_training=False,
                                transform=BaseTransform(cfg.val_input_size, cfg.rgb_mean, cfg.rgb_std))
    elif cfg.dataset == 'TD500':
        train_dataset = TD500Text(cfg.train_data_root, is_training=True,
                                transform=Augmentation(cfg.train_input_size, cfg.train_rgb_mean, cfg.train_rgb_std))
        val_dataset = TD500Text(cfg.val_data_root, is_training=False,
                                transform=BaseTransform(cfg.val_input_size, cfg.val_rgb_mean, cfg.val_rgb_std)) #TODO cfg.test_size should be cfg.val_input_size
    else:
        raise NotImplementedError

    # Create the dataloader
    train_loader = data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
                                    shuffle=cfg.train_shuffle, num_workers=cfg.train_num_workers) #, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=cfg.val_batch_size,
                                    shuffle=cfg.val_shuffle, num_workers=cfg.val_num_workers) #, pin_memory=True)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
    #                            weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)

    # Create the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train_step_size, gamma=cfg.train_gamma)

    # Create the loss criterion
    criterion = TextLoss()

    # Load the pretrained model
    if cfg.global_pretrain:
        model.load_model(cfg.pretrain_model)
        print('\nSuccesfully loaded pretrained model from: "{}"\n'.format(cfg.pretrain_model))

    # Move the model to GPU
    if cfg.use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Start training
    print('Device: {}'.format(cfg.device))
    print('Initializing Training Dataset from: "{}"'.format(os.path.join(os.path.join('data',cfg.train_data_root),'Train')))
    print('Training DataLoader has {} iterations\n'.format(int(len(train_loader))))
    print('Initializing Validation Dataset from: "{}"'.format(os.path.join(os.path.join('data',cfg.val_data_root),'Test')))
    print('Validation DataLoader has {} iterations\n'.format(int(len(val_loader))))
    print('An Evaluation will run every {} iteration\n'.format(cfg.val_freq))

    for epoch in range(cfg.epochs):
        scheduler.step()
        train(model, train_loader , criterion, scheduler, optimizer, epoch, cfg, val_loader)

        if epoch % cfg.save_freq == 0:
            save_name = os.path.join(cfg.save_path, '{}_{}_{}.pth'.format(cfg.dataset, cfg.net, epoch))
            torch.save(model.state_dict(), save_name)
    
    # Save the final model
    save_name = os.path.join(cfg.save_path, '{}_{}_{}.pth'.format(cfg.dataset, cfg.net, cfg.epochs))
    torch.save(model.state_dict(), save_name)

    print('Training is finished.')



def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')



def inference(model, test_loader, output_dir, cfg):

    total_time = 0.
    if cfg.exp_name != "MLT2017":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()

        input_dict['img'] = to_device(image)
        # get detection result
        start = time.time()
        # @MM TODO torch.cuda.synchronize()
        output_dict = model(input_dict)
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0
        idx = 0  # test mode can only run with batch_size == 1
        print('detect {} / {} images: {}.'.format(i + 1, len(test_loader), meta['image_id'][idx]))

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()

        gt_contour = []
        label_tag = meta['label_tag'][idx].int().cpu().numpy()
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())

        gt_vis = visualize_gt(img_show, gt_contour, label_tag)

        show_map = np.concatenate([heat_map, gt_vis], axis=1)
        show_map = cv2.resize(show_map, (320 * 3, 320))
        im_vis = np.concatenate([show_map, show_boundary], axis=0)

        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
        cv2.imwrite(path, im_vis)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "MLT2017":

            out_dir = os.path.join(output_dir, str(cfg.checkepoch))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            data_transfer_MLT2017(contours, os.path.join(out_dir, fname))
        elif cfg.exp_name == "TD500":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            data_transfer_TD500(contours, os.path.join(output_dir, fname))

        else:
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))


def train(model, train_loader, criterion, scheduler, optimizer, epoch, cfg, val_loader):
    '''
    Define the training function for our model
    '''
    global train_step
    losses_meters = {'total_loss': AverageMeter(),
                    'cls_loss': AverageMeter(),
                    'distance loss': AverageMeter(),
                    'point_loss': AverageMeter()}
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    scheduler.step()
    end = time.time()

    for i, (img, train_mask, tr_mask, distance_field,
            direction_field, weight_matrix, gt_points,
            proposal_points, ignore_tags) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = to_device(img)
        
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, gt_points, \
        proposal_points, ignore_tags = to_device(img, 
                                                train_mask, tr_mask, distance_field,
                                                direction_field, weight_matrix, gt_points,
                                                proposal_points, ignore_tags)
        input_dict = dict()
        input_dict['img'] = to_device(img)
        input_dict['train_mask'] = to_device(train_mask)
        input_dict['tr_mask'] = to_device(tr_mask)
        input_dict['distance_field'] = to_device(distance_field)
        input_dict['direction_field'] = to_device(direction_field)
        input_dict['weight_matrix'] = to_device(weight_matrix)
        input_dict['gt_points'] = to_device(gt_points)
        input_dict['proposal_points'] = to_device(proposal_points)
        input_dict['ignore_tags'] = to_device(ignore_tags)
        output = model(input_dict)

        lossT = criterion(input_dict, output)
        if cfg.wandb_flag:
            wandb.log(lossT)

        loss = lossT['total_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for loss_key_tmp in lossT.keys():
            losses_meters[loss_key_tmp].update(lossT[loss_key_tmp].item())

        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfg.print_freq == 0:
            print('Epoch [{}/{}] Step [{}/{}]: total_loss: {:.4f}({:.4f}), cls_loss: {:.4f}({:.4f}), '
            'distance_loss: {:.4f}({:.4f}), point_loss: {:.4f}({:.4f}), Batch Time: {:.4f}({:.4f}), '
            'Data Time: {:.4f}({:.4f}), lr: {:.4f}'.format(epoch, cfg.epochs, i + 1, len(train_loader), losses_meters['total_loss'].val, losses_meters['total_loss'].avg, 
                                                        losses_meters['cls_loss'].val, losses_meters['cls_loss'].avg, losses_meters['distance loss'].val, losses_meters['distance loss'].avg, 
                                                        losses_meters['point_loss'].val, losses_meters['point_loss'].avg, batch_time.val, batch_time.avg, data_time.val, 
                                                        data_time.avg, scheduler.get_lr()[0]))
        if (i + 1) % cfg.val_freq == 0:
            print("Validation Epoch [{}/{}] Step [{}/{}]: ...".format(epoch, cfg.epochs, i + 1, len(val_loader)))
            losses_meters_val,batch_time_val, data_time_val = validation(model, val_loader, criterion, cfg)
            print('\tval_total_loss: {:.4f}({:.4f}), val_cls_loss: {:.4f}({:.4f}), '
            'val_distance_loss: {:.4f}({:.4f}), val_point_loss: {:.4f}({:.4f}), val_Batch Time: {:.4f}({:.4f}), '
            'val_Data Time: {:.4f}({:.4f})'.format(losses_meters_val['total_loss'].val, losses_meters_val['total_loss'].avg, 
                                                losses_meters_val['cls_loss'].val, losses_meters_val['cls_loss'].avg, losses_meters_val['distance loss'].val, losses_meters_val['distance loss'].avg, 
                                                losses_meters_val['point_loss'].val, losses_meters_val['point_loss'].avg, batch_time_val.val, batch_time_val.avg, data_time_val.val, 
                                                data_time_val.avg))
        

def validation(model, val_loader, criterion, cfg):
    with torch.no_grad():
        model.eval()
    losses_meters_val = {'val_total_loss': AverageMeter(),
                    'val_cls_loss': AverageMeter(),
                    'val_distance loss': AverageMeter(),
                    'val_point_loss': AverageMeter()}
    batch_time_val = AverageMeter()
    data_time_val = AverageMeter()
    end = time.time()

    for i, (img, train_mask, tr_mask, distance_field,
            direction_field, weight_matrix, gt_points,
            proposal_points, ignore_tags) in enumerate(val_loader):
        data_time_val.update(time.time() - end)
        img = to_device(img)
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, gt_points, \
        proposal_points, ignore_tags = to_device(img, 
                                                train_mask, tr_mask, distance_field,
                                                direction_field, weight_matrix, gt_points,
                                                proposal_points, ignore_tags)
        input_dict = dict()
        input_dict['img'] = to_device(img)
        input_dict['train_mask'] = to_device(train_mask)
        input_dict['tr_mask'] = to_device(tr_mask)
        input_dict['distance_field'] = to_device(distance_field)
        input_dict['direction_field'] = to_device(direction_field)
        input_dict['weight_matrix'] = to_device(weight_matrix)
        input_dict['gt_points'] = to_device(gt_points)
        input_dict['proposal_points'] = to_device(proposal_points)
        input_dict['ignore_tags'] = to_device(ignore_tags)
        output = model(input_dict)

        lossT = criterion(input_dict, output)
        for loss_key_tmp in lossT.keys():
            losses_meters_val['val_'+loss_key_tmp].update(lossT[loss_key_tmp].item())

        batch_time_val.update(time.time() - end)
        end = time.time()
        if cfg.wandb_flag:
            wandb.log(lossT)
    return losses_meters_val,batch_time_val, data_time_val


# if (i +1) % cfg.val_freq == 0:
#    inference(model, val_loader, 'out_temp', cfg)
#    if cfg.exp_name == "Totaltext":
#        deal_eval_total_text(debug=True)
#    elif cfg.exp_name == "Ctw1500":
#        deal_eval_ctw1500(debug=True)
#    elif cfg.exp_name == "Icdar2015":
#        deal_eval_icdar15(debug=True)
#    elif cfg.exp_name == "TD500":
#       deal_eval_TD500(debug=True)
#    else:
#        print("{} is not justify".format(cfg.exp_name))
#
# if (i + 1) % cfg.save_freq == 0:
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'scheduler': scheduler.state_dict(),
#         'cfg': cfg,
#         'loss': loss_meter.avg,
#         'losses_meters': losses_meters,
#         'train_step': train_step
#     }, cfg.ckpt_dir, 'checkpoint_{}.pth'.format(train_step))
#     train_step += 1


if __name__ == '__main__':
    main()
