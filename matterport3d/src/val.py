from pickle import TRUE
from random import randint
from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import calc_quanti, to_numpy
import model.ops as ops
import model.models as m
import shutil

opt = Options(sys.argv[0])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_eval(out_dir, mse, psnr, ssim):
    f = open(os.path.join(out_dir,'eval.txt'), 'w')
    f.write('MSE: ' + str(mse) + '\n')
    f.write('PSNR: ' + str(psnr) + '\n')
    f.write('SSIM: ' + str(ssim) + '\n')
    f.close()

def estimate_fov_new(model_name, data_num, offset, output_dir, gt_data_path):
    folder_name = 'pano_' + str(data_num)
    
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
        print(impath)
        imglist.append(read_img_to_tensor(impath))


    img_tensor = torch.cat([imglist[0], imglist[1], imglist[2], imglist[3]], dim=2)
    in_img = make_input_batch(img_tensor, device)
    in_img = ops.downsample(in_img,1)#2
    _, fov_out = net(in_img)
   
    fov = torch.argmax(fov_out)
    #fov_pred = fov.item() * 2
    fov_pred = 64 * 2 # soojie: 45-degree 
    #fov_pred = 85 * 2 # soojie: 60-degree 
    #fov_pred = 106 * 2 # soojie: 75-degree 

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
    
    h,w,_ = imglist[0].shape
    up = np.zeros((h, w, 3), np.uint8)
    down = np.zeros((h,w, 3), np.uint8)
    cv2.imwrite(os.path.join(output_dir,'fov\\posy.jpg'), up)
    cv2.imwrite(os.path.join(output_dir,'fov\\negy.jpg'), down)

    img_horiz = np.hstack([imglist[0], imglist[1], imglist[2], imglist[3]])
    
    img_out ='img_out.jpg'
    img_cat = np.vstack([np.zeros_like(img_horiz), img_horiz, np.zeros_like(img_horiz)])
    img_cat = utl.numpy_to_pano(img_cat)
    outpath = os.path.join(output_dir,folder_name, img_out)
    cv2.imwrite(outpath, img_cat)
    
    return os.path.join(output_dir,folder_name), img_out

def panorama_generate(folder, fov_out, model_name, net_type=None):
    # Init network
    generator = m.GM().to(device)    
    generator.eval()
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    
    model = torch.load(model_path)
    generator.load_state_dict(model['Generator'], strict=True)

    # Init input
    in_img = read_img_to_tensor(os.path.join(folder,fov_out))
    in_img = torch.unsqueeze(in_img, 0)
    in_img = in_img.to(device)
    in_img = ops.downsample(in_img)

    if net_type == 'small':
        out_s = generator(in_img)
        out = out_s
    elif net_type == 'medium':
        out_s, out_m = generator(in_img)
        out = out_m
    elif net_type == 'large':
        out_s, out_m, out_l = generator(in_img)
        out = out_l

    save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out)
    print("finished saving image")


def panorama_generate_val(folder, fov_out,folder_2, fov_out_2, generator, net_type='small'):
    
    # Init input
    in_img = read_img_to_tensor(os.path.join(folder,fov_out))
    in_img = torch.unsqueeze(in_img, 0)
    in_img = in_img.to(device)
    in_img = ops.downsample(in_img)
    in_img_2 = read_img_to_tensor(os.path.join(folder_2,fov_out_2))
    in_img_2 = torch.unsqueeze(in_img_2, 0)
    in_img_2 = in_img_2.to(device)
    in_img_2 = ops.downsample(in_img_2)

    out_s = fov_out
    out_m = fov_out
    out_l = fov_out

    if net_type == 'small':
        out_s = generator(in_img,in_img_2)
        save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out_s)
        
    if net_type == 'medium':
        out_s, out_m = generator(in_img,in_img_2)
        save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out_m)
    
    if net_type == 'large':
        out_s, out_m, out_l = generator(in_img,in_img_2)
        save_img_from_tensor(os.path.join(folder,'pano_out.jpg'), out_l)

    print("finished saving image")

def read_img_to_tensor(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def make_input_batch(img_tensor, device, down=2):
    in_img = torch.unsqueeze(img_tensor, 0)
    in_img = in_img.to(device)
    return in_img

def generate_pad_img(img, fov):
    pad = int((256-fov)/2)

    x = int((256-fov)/2)
    y = int((256-fov)/2)
    w = fov
    h = fov
    img = cv2.resize(img, (256,256))
    img = img[y:y+h,x:x+w]

    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    return img

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

def valdiate(generator, net_type, test=False):
    outdir = '..\\output'
    outdir_2 = '..\\output_2'

    gt_dir = '.\\3d60_dataset_depth_train' 
    gt_dir_2 = '.\\3d60_dataset_rgb_train' 
    
    if type == 'rgb':
        gt_dir = '.\\3d60_dataset_rgb_train' 
        gt_dir_2 = '.\\3d60_dataset_depth_train' 

    data_len = 1
    #data_offset = randint(0,8718)
    data_offset = 0

    if test == True:
        #data_offset = randint(8729,10900)
        data_offset = 0

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(outdir_2):
        os.makedirs(outdir_2)

    # model: matterport rgb,depth
    for i in range(0, data_len):
        folder,img = estimate_fov_new('model_191105_fov_49', i,data_offset, outdir,gt_dir)
        folder_2,img_2 = estimate_fov_new('model_191105_fov_49', i,data_offset, outdir_2,gt_dir_2)
        panorama_generate_val(folder,img,folder_2,img_2,generator, net_type)

    mse, psnr, ssim = evaluate_output(gt_dir,outdir, data_len, data_offset)
    avg_psnr = np.average(psnr)
    avg_mse = np.average(mse)
    avg_ssim = np.average(ssim)

    save_eval(outdir, avg_mse, avg_psnr, avg_ssim)

    print('PSNR :',avg_psnr)
    print('MSE :',avg_mse)
    print('SSIM :',avg_ssim)

    return avg_psnr, avg_ssim, avg_mse

def rmdir():
    if os.path.isdir('..\\output'):
        shutil.rmtree('..\\output')

