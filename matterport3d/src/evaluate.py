from pickle import TRUE
from util.base import *
from util.opt import Options
from util.utilities import calc_quanti, to_numpy
import util.utilities as utl
import model.models as m
import model.ops as ops
import model.models as m

opt = Options(sys.argv[0])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_eval(out_dir, mse, psnr, ssim):
    f = open(os.path.join(out_dir,'eval.txt'), 'w')
    f.write('MSE: ' + str(mse) + '\n')
    f.write('PSNR: ' + str(psnr) + '\n')
    f.write('SSIM: ' + str(ssim) + '\n')
    f.close()

def estimate_fov_new(model_name, data_num, offset, output_dir, gt_data_path, net_type):
    folder_name = 'pano_' + str(data_num) # 저장폴더이름 
    pano_gt_name = os.path.join(gt_data_path,'pano_' + str(data_num+offset),'pano_' + str(data_num+offset) + '.jpg')
   
    if not os.path.isdir(os.path.join(output_dir,'fov')):
        os.makedirs(os.path.join(output_dir,'fov'))
    
    if not os.path.isdir(os.path.join(output_dir,folder_name)):
        os.makedirs(os.path.join(output_dir,folder_name))

    # Init network
    net = m.FOVnetwork().to(device)
    net.eval()

    # load pretrained model
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    model = torch.load(model_path)
    net.load_state_dict(model, False)

    imglist = []
    for i in range(4):
        
        im = ''
        if i == 0: im = 'gt_3'
        if i == 1: im = 'gt_1'
        if i == 2: im = 'gt_4'
        if i == 3: im = 'gt_2'
        impath = os.path.join(gt_data_path,'pano_'+str(data_num + offset), im + '.jpg')
        
        imglist.append(read_img_to_tensor(impath))

    img_tensor = torch.cat([imglist[0], imglist[1], imglist[2], imglist[3]], dim=2)
    in_img = make_input_batch(img_tensor, device)
    in_img = ops.downsample(in_img,1)#2
    _, fov_out = net(in_img)
   
    fov = torch.argmax(fov_out)
    print('fov : ' + str(fov))
    fov_pred = fov.item() * 2
    #fov_pred = 128 * 2 # soojie: 90-degree 
    #fov_pred = 64 * 2 # soojie: 45-degree 
    fov_pred = 85 * 2 # soojie: 60-degree 
    #fov_pred = 106 * 2 # soojie: 75-degree 
    #fov_pred = 242 # soojie: 85-degree

    imglist = []
    for i in range(4):
        
        im = ''
        if i == 0: im = 'gt_3'
        if i == 1: im = 'gt_1'
        if i == 2: im = 'gt_4'
        if i == 3: im = 'gt_2'
        impath = os.path.join(gt_data_path,'pano_'+str(data_num + offset), im + '.jpg')
        
        img = cv2.imread(impath)
        fov = generate_pad_img(img,fov_pred)
        imglist.append(fov)
        cv2.imwrite(os.path.join(output_dir,'fov\\gt_'+str(4-i)+'.jpg'),fov)
    
    img_horiz = np.hstack([imglist[0], imglist[1], imglist[2], imglist[3]])
    
    img_out ='img_out.jpg'
    img_cat = np.vstack([np.zeros_like(img_horiz), img_horiz, np.zeros_like(img_horiz)])
    img_cat = utl.numpy_to_pano(img_cat)

    #mask = make_mask(fov_pred)
    mask = make_mask(212)

    if net_type == 'small': 
        img_cat = cv2.resize(img_cat,dsize=(256,128))
        mask = cv2.resize(mask,dsize=(256,128))
    if net_type == 'medium': 
        img_cat = cv2.resize(img_cat,dsize=(512,256))
        mask = cv2.resize(mask,dsize=(512,256))
    if net_type == 'large': 
        img_cat = cv2.resize(img_cat,dsize=(1024,512))
        mask = cv2.resize(mask,dsize=(1024,512))

    outpath = os.path.join(output_dir,folder_name, img_out)
    cv2.imwrite(outpath, img_cat)

    return os.path.join(output_dir,folder_name), img_out, pano_gt_name,mask

def panorama_generate(folder, fov_out,folder2,fov_out2, pano_gt, model_name,model_name_2, net_type=None, mask = None):
    # Init network
    generator = None
    generator_2 = None

    if net_type == 'small':
        generator = m.GS().to(device)
        generator_2 = m.GS().to(device)
    elif net_type == 'medium':
        generator = m.GM().to(device)
        generator_2 = m.GM().to(device)
    elif net_type == 'large':
        generator == m.GL2.to(device)
        generator_2 == m.GL2.to(device) 
            
    generator.eval()
    generator_2.eval()
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    model_path_2 = os.path.join(opt.model_path, model_name_2 + '.pt')
    
    model = torch.load(model_path)
    model_2 = torch.load(model_path_2)
    generator.load_state_dict(model['Generator'], strict=True)
    generator_2.load_state_dict(model_2['Generator'], strict=True)

    # Init input
    in_img = read_img_to_tensor(im_path = os.path.join(folder,fov_out))
    in_img = torch.unsqueeze(in_img, 0)
    in_img = in_img.to(device)
    #in_img = ops.downsample(in_img)

    # Init input
    #in_img2 = read_img_to_tensor(os.path.join(folder2,fov_out2))
    #in_img2 = torch.unsqueeze(in_img2, 0)
    #in_img2 = in_img2.to(device)

    pano_pair = read_img_to_tensor(im_path=pano_gt,tag=net_type)
    pano_pair = torch.unsqueeze(pano_pair, 0)
    pano_pair = pano_pair.to(device)

    mask = read_img_to_tensor(mask=mask,tag=net_type)
    mask = torch.unsqueeze(mask, 0)
    mask = mask.to(device)

    if net_type == 'small':
        out_s_2, feature = generator_2(pano_pair*mask)
        out_s,_ = generator(in_img, feature, tag='small')
        out = out_s
    elif net_type == 'medium':
        out_s_2, out_m_2, feature = generator_2(pano_pair*mask)
        out_s, out_m, _ = generator(in_img,feature, tag='medium')
        out = out_m
    elif net_type == 'large':
        out_s_2, out_m_2, out_l_2, feature = generator_2(pano_pair*mask)
        out_s, out_m, out_l, _ = generator(in_img, feature, tag='large')
        out = out_l

    #show_img_from_tensor(out_s_2)
    save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out)
    print("finished saving image")

def panorama_generate_val(folder, fov_out, generator, net_type='small'):
    
    # Init input
    in_img = read_img_to_tensor(os.path.join(folder,fov_out))
    in_img = torch.unsqueeze(in_img, 0)
    in_img = in_img.to(device)
    in_img = ops.downsample(in_img)

    out_s = fov_out
    out_m = fov_out
    out_l = fov_out

    if net_type == 'small':
        out_s = generator(in_img)
        save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out_s)
        
    if net_type == 'medium':
        out_s, out_m = generator(in_img)
        save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out_m)
    
    if net_type == 'large':
        out_s, out_m, out_l = generator(in_img)
        save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out_l)

    print("finished saving image")

def make_mask(fov, scale=1.):
        empty = np.zeros((256,256,3), np.uint8)
        f = generate_pad_img(empty,fov,pad_type='ones')
        empty_horiz = np.hstack([f,f,f,f])
        empty_cat = np.vstack([np.ones_like(empty_horiz), empty_horiz, np.ones_like(empty_horiz)])
        empty_cat = utl.numpy_to_pano(empty_cat)
        name = '.\\mask.jpg'
        cv2.imwrite(name, empty_cat)
        #img = cv2.imread('.\\mask.jpg')
        img_rsz = cv2.resize(empty_cat, (0,0), fx=scale, fy=scale)
        return img_rsz

def read_img_to_tensor(im_path=None, mask=None, tag=None):
    
    if im_path!= None:
        im = cv2.imread(im_path)
    else: im = mask
    print(im_path)
    if tag != None:
            if tag == 'small':
                im = cv2.resize(im,dsize=(256,128))
            if tag == 'medium':
                im = cv2.resize(im,dsize=(512,256))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im_path != None:
        im = im / 127.5 - 1
    
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def make_input_batch(img_tensor, device, down=2):
    in_img = torch.unsqueeze(img_tensor, 0)
    in_img = in_img.to(device)
    return in_img

def generate_pad_img(img, fov, pad_type='zeros'):
    pad = int((256-fov)/2)

    x = int((256-fov)/2)
    y = int((256-fov)/2)
    w = fov
    h = fov
    img = cv2.resize(img, (256,256))
    img = img[y:y+h,x:x+w]

    if pad_type == 'zeros':
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    if pad_type == 'ones':
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, [1,1,1])
    return img

def show_img_from_tensor(img_tensor):
    img = to_numpy(img_tensor) * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('a.jpg', img.astype(np.uint8))
    im = cv2.imread('a.jpg')
    cv2.imshow('aa',im)
    cv2.waitKey(0)

def save_img_from_tensor(im_path, img_tensor):
    img = to_numpy(img_tensor) * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path, img.astype(np.uint8))


def evaluate_output(path_gt, path_out, data_len, offset):
    total_mse = []
    total_psnr = []
    total_ssim = []
    
    f = open(os.path.join(path_out,'result.txt'), 'w')

    max_psnr = [0,-1]
    max_ssim = [0,-1]
    min_psnr = [100,-1]
    min_ssim = [1,-1]

    for i in range(0,data_len):
        print('Processing data %d/%d' %(i, data_len))
        n=offset
        folder_gt = 'pano_' + str(i + n)
        folder_out = 'pano_' + str(i)
        gt_im_path = os.path.join(path_gt,folder_gt, 'pano_' + str(i + n) + '.jpg')
        out_im_path = os.path.join(path_out,folder_out, 'pano_out.jpg')
        
        print(gt_im_path, out_im_path)
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)

        gt_im = cv2.resize(gt_im, dsize=(512,256))
        out_im = cv2.resize(out_im, dsize=(512,256))
        
        mse, psnr, ssim = calc_quanti(out_im, gt_im)
        print(mse, psnr, ssim)
        total_mse.append(mse)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
        f.write('['+str(i)+']:'+str(i)+' , ' + str(i+n) + '\n')
        f.write('   MSE: ' + str(mse) + '\n')
        f.write('   PSNR: ' + str(psnr) + '\n')
        f.write('   SSIM: ' + str(ssim) + '\n')
        
        if psnr < min_psnr[0]:
            min_psnr[0] = psnr
            min_psnr[1] = i
        if psnr > max_psnr[0]:
            max_psnr[0] = psnr
            max_psnr[1] = i

        if ssim < min_ssim[0]:
            min_ssim[0] = ssim
            min_ssim[1] = i
        if ssim > max_ssim[0]:
            max_ssim[0] = ssim
            max_ssim[1] = i

    f.write('\nMin PSNR: ' + str(min_psnr[0]) + ' , pano_' + str(min_psnr[1]))
    f.write('\nMax PSNR: ' + str(max_psnr[0]) + ' , pano_' + str(max_psnr[1]))
    f.write('\nMin SSIM: ' + str(min_ssim[0]) + ' , pano_' + str(min_ssim[1]))
    f.write('\nMax SSIM: ' + str(max_ssim[0]) + ' , pano_' + str(max_ssim[1]))
    f.close()
    return total_mse, total_psnr, total_ssim

def valdiate(self, generator, net_type, test=True):
    outdir = '..\\output'
    gt_dir = '.\\matterport_rgb_data_train' 
    data_len = 10
    data_offset = 0

    if test == True:
        data_offset = 8729

    for i in range(0, data_len):
        folder,img, pano_gt,mask = estimate_fov_new('model_210713_fov_1343', i,data_offset, outdir,gt_dir)
        panorama_generate_val(folder,img,generator, net_type)

    mse, psnr, ssim = evaluate_output(gt_dir,outdir, data_len, data_offset)
    avg_psnr = np.average(psnr)
    avg_mse = np.average(mse)
    avg_ssim = np.average(ssim)

    save_eval(outdir, avg_mse, avg_psnr, avg_ssim)

    print('PSNR :',avg_psnr)
    print('MSE :',avg_mse)
    print('SSIM :',avg_ssim)

    return avg_psnr, avg_ssim, avg_mse


outdir = '..\\output'
outdir_depth = '..\\output_depth'

#depth = 'D:\\work\\Data\\3d60_dataset_depth_test'
#gt_dir = 'D:\\work\\Data\\3d60_dataset_rgb_test'
#gt_dir_depth = 'D:\\work\\Data\\3d60_dataset_rgb_test'

gt_dir = 'D:\\work\\Data\\3d60_dataset_rgb_train'
gt_dir_depth = 'D:\\work\\Data\\3d60_dataset_depth_train'

#depth = 'D:\\work\\Data\\hohonet_depth'
#gt_dir = 'D:\\work\\Data\\hohonet_rgb'
#gt_dir_depth = 'D:\\work\\Data\\hohonet_rgb'

net_type = 'small'

data_len=500
data_offset=8030

#data_len = 1759
#data_offset = 0

#data_len = 272
#data_offset = 1760

#data_len = 1612
#data_offset = 2032

for i in range(0, data_len):
    folder,img,pano_gt, mask = estimate_fov_new('D:\\work\\trained_model\\model_191105_fov_49', i,data_offset, outdir,gt_dir,net_type=net_type)
    folder_depth,img_depth,pano_gt_depth, mask_depth = estimate_fov_new('D:\\work\\trained_model\\model_191105_fov_49', i,data_offset, outdir_depth,gt_dir_depth, net_type=net_type)
    panorama_generate(folder,img,folder_depth,img_depth,pano_gt_depth,'C:\\Users\\sujie\\Desktop\\a100_new\\new_model\\new\\suncg_fusion_1_rgb_212_82', 'C:\\Users\\sujie\\Desktop\\a100_new\\new_model\\new\\suncg_fusion_1_depth_212_82', net_type=net_type,mask=mask_depth)
    panorama_generate(folder_depth,img_depth,folder,img,pano_gt,'C:\\Users\\sujie\\Desktop\\a100_new\\new_model\\new\\suncg_fusion_1_depth_212_82', 'C:\\Users\\sujie\\Desktop\\a100_new\\new_model\\new\\suncg_fusion_1_rgb_212_82',net_type=net_type, mask=mask)

mse, psnr, ssim = evaluate_output(gt_dir,outdir, data_len, data_offset)
mse_depth, psnr_depth, ssim_depth = evaluate_output(gt_dir_depth,outdir_depth, data_len, data_offset)

avg_psnr = np.average(psnr)
avg_mse = np.average(mse)
avg_ssim=np.average(ssim)

avg_psnr_depth = np.average(psnr_depth)
avg_mse_depth = np.average(mse_depth)
avg_ssim_depth = np.average(ssim_depth)

save_eval(outdir, avg_mse, avg_psnr, avg_ssim)
save_eval(outdir_depth, avg_mse_depth, avg_psnr_depth, avg_ssim_depth)

print('RGB-PSNR :',avg_psnr)
print('RGB-MSE :',avg_mse)
print('RGB-SSIM :',avg_ssim)
print('RGBD-PSNR :',avg_psnr_depth)
print('RGBD-MSE :',avg_mse_depth)
print('RGBD-SSIM :',avg_ssim_depth)

