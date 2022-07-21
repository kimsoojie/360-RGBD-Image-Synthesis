from util.base import *
from util.opt import Options
from dataset.dataset import Dataset
from model.Networks import Networks
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    opt = Options(sys.argv[0])

    # Define Dataset
    # =================
    data_loader = Dataset('train',resize=opt.resize).load_data()
    data_loader_depth = Dataset('train_depth',resize=opt.resize).load_data_depth() 
    
    net = Networks(device, opt.net, loss='lsgan',type='rgb')
    net.set_phase('train')
    net.print_structure()

    #soojie: depth
    net_depth = Networks(device, opt.net, loss='lsgan',type='depth')
    net_depth.set_phase('train')
    net_depth.print_structure()

    #load model
    #net.load_model('checkpoint_0_110_small', False)
    #net_depth.load_model('checkpoint_1_110_small', False)
    
    if opt.net == 'small':
        forward_call = net.train_small
        txt_summary_call = net.print_summary_small
        #soojie: depth
        forward_call_depth = net_depth.train_small
        txt_summary_call_depth = net_depth.print_summary_small
    elif opt.net == 'medium':
        forward_call = net.train_medium
        txt_summary_call = net.print_summary_small
        #soojie: depth
        forward_call_depth = net_depth.train_medium
        txt_summary_call_depth = net_depth.print_summary_medium
    elif opt.net == 'large':
        forward_call = net.train_large
        txt_summary_call = net.print_summary_small

    total_step = 0
    start = time.time()
    for epoch in range(opt.total_epochs):
        step = 0

        net.save_ckpt(0)
        net_depth.save_ckpt(1)
        print('total step: ',total_step)

        if epoch == 100 and step == 0:
            net.save_model(epoch, 'model_rgb_100_211111_' + str(total_step)+'_' + opt.net)
            net_depth.save_model(epoch, 'model_depth_100_211111_'+str(total_step)+'_' + opt.net)

      
        for item, item_depth in zip(data_loader, data_loader_depth):
            in_img, gt_img, gt_fov, mask = item['input'], item['gt'], item['fov'], item['mask']
            in_img_depth, gt_img_depth, gt_fov_depth, mask_depth = item_depth['input'], item_depth['gt'], item_depth['fov'], item_depth['mask']
            
            if in_img.size()[0] != opt.train_batch:
                break

            if in_img_depth.size()[0] != opt.train_batch:
                break

            # Load image to network
            net.load_input_batch(in_img, gt_img, gt_fov, mask, in_img_depth, gt_img_depth, gt_fov_depth, mask_depth)
            net_depth.load_input_batch(in_img_depth, gt_img_depth, gt_fov_depth,mask_depth, in_img, gt_img, gt_fov, mask)

            # Forward network
            forward_call(total_step,epoch,'rgb', net_depth.Generator)
            forward_call_depth(total_step,epoch, 'depth', net.Generator)

            end = time.time()
            elapsed = end - start

            # Print network loss
            # Add Tensorboard summary
            if step % 50 == 0:
                txt_summary_call(epoch, step,'rgb') 
                txt_summary_call_depth(epoch, step,'depth')
                print('Time elapsed', elapsed, 'seconds')
           
            step += 1
            total_step += 1

    net.save_model(epoch, 'model_rgb_200_211111_'+str(total_step)+'_' + opt.net)
    net_depth.save_model(epoch, 'model_depth_200_211111_'+str(total_step)+'_' + opt.net)


if __name__ == '__main__':
    main()
