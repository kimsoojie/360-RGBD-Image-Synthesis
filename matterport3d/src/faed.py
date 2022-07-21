from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import calc_quanti, to_numpy
import model.ops as ops
import model.models as m
from scipy import linalg

opt = Options(sys.argv[0])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_img_to_tensor(in_img):
    im = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    im = im / 255
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    #print(np.shape(tsr)) #3*128*256
    return tsr

def load_model(model, model_pair, net_type):
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
    model_path = os.path.join(model + '.pt')
    model_path_2 = os.path.join(model_pair + '.pt')
    
    model = torch.load(model_path)
    model_2 = torch.load(model_path_2)
    generator.load_state_dict(model['Generator'], strict=True)
    generator_2.load_state_dict(model_2['Generator'], strict=True)

    return generator, generator_2


def generator(tag, model_name, model_name_pair, net_type):
    # Init network
    if net_type == 'small': 
        g = m.GS_feature(tag, False).to(device) 
        g_p = m.GS_feature(tag, True).to(device) 
    elif net_type == 'medium':
        g = m.GM().to(device)
    elif net_type == 'large':
        g = m.GL2().to(device)
   
    g.eval()
    g_p.eval()
    model_path_rgb = model_name + '.pt'
    model_path_depth = model_name_pair + '.pt'

    model_rgb = torch.load(model_path_rgb)
    g.load_state_dict(model_rgb['Generator'], strict=True)

    model_depth = torch.load(model_path_depth)
    g_p.load_state_dict(model_depth['Generator'], strict=True)

    return g, g_p

def preprocessing(image):
    in_img = read_img_to_tensor(image)
    in_img = torch.unsqueeze(in_img, 0)
    #print(np.shape(in_img)) [1,3,128,256]
    in_img = in_img.to(device)
    return in_img

def calc_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calc_frechet_distance(mu1, sigma1, mu2, sigma2,eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean



def longitudinal_invariant_feature(feature_map, height):
    # channel: [0,1023]
    # height: [0,feature map height-1]

    f = feature_map
    h,w = np.shape(f)

    val = 0
    for i in range(0,w):
        val = val + f[height,i]
    val = val / w
    return val

def latitudinal_equivariance(features):
    v = []
    
    for c in range(0,2048):
        #f_name = 'f_'+str(c)+'.jpg'
        #f = cv2.imread(os.path.join(feature_dir,f_name),0)
        f = features[c]
        h,w = np.shape(f)

        db=1.0*math.pi/float(h)
        b=-0.5*math.pi

        val = 0.0
        for i in range(0,h):
            val = val + math.cos(b)*longitudinal_invariant_feature(f,i)
            b = b + db
        v.append(val)
    
    return v

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

def generate_random_fov():
    fov_range = np.arange(128,212,2)
    np.random.shuffle(fov_range)
    return fov_range[0]

def make_mask(fov, pad_type = 'ones'):
    if pad_type == 'ones':
        empty = np.zeros((256,256,3), np.uint8)
    if pad_type == 'zeros':
        empty = np.ones((256,256,3), np.uint8)

    f = generate_pad_img(empty,fov,pad_type=pad_type)
    empty_horiz = np.hstack([f,f,f,f])
    if pad_type == 'ones':
        empty_cat = np.vstack([np.ones_like(empty_horiz), empty_horiz, np.ones_like(empty_horiz)])
    if pad_type == 'zeros':
        empty_cat = np.vstack([np.zeros_like(empty_horiz), empty_horiz, np.zeros_like(empty_horiz)])
    empty_cat = utl.numpy_to_pano(empty_cat)
    #name = '.\\mask.jpg'
    #cv2.imwrite(name, empty_cat)
    #img = cv2.imread('.\\mask.jpg')
    img_rsz = cv2.resize(empty_cat, dsize=(256,128))
    return img_rsz

def main():
    model_path_rgb = 'C:\\Users\\sujie\\Desktop\\a100_new\\new_model\\new\\matterport3d_fusion_1_rgb_212_77_faed'
    model_path_depth = 'C:\\Users\\sujie\\Desktop\\a100_new\\new_model\\new\\matterport3d_fusion_1_depth_212_77_faed'
    outdir_rgb = 'D:\\work\\faed\\matterport3d_2\\output'
    outdir_depth = 'D:\\work\\faed\\matterport3d_2\\output_depth'
    outdir_rgb_stanford3d = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\stanford3d\\fusion_1\\output'
    outdir_depth_stanford3d = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\stanford3d\\fusion_1\\output_depth'
    outdir_rgb_suncg = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\suncg\\fusion_1\\output'
    outdir_depth_suncg = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\suncg\\fusion_1\\output_depth'
    
    gt_dir_rgb = 'D:\\work\\Data\\3d60_dataset_rgb_test'
    gt_dir_depth = 'D:\\work\\Data\\3d60_dataset_depth_test'
    blur_dir_rgb = 'C:\\Users\\sujie\\Desktop\\test\\blur_rgb'
    blur_dir_depth = 'C:\\Users\\sujie\\Desktop\\test\\blur_depth'
    salt_dir_rgb = 'C:\\Users\\sujie\\Desktop\\test\\salt_rgb'
    salt_dir_depth = 'C:\\Users\\sujie\\Desktop\\test\\salt_depth'
    single_rgb = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\matterport3d\\single_rgb\\output_new4'
    single_depth = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\matterport3d\\single_depth\\output'
    single_rgb_stanford3d = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\stanford3d\\single_rgb\\output'
    single_depth_stanford3d = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\stanford3d\\single_depth\\output'
    single_rgb_suncg = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\suncg\\single_rgb\\output'
    single_depth_suncg = 'C:\\Users\\sujie\\Desktop\\a100_new\\windows_ver\\suncg\\single_depth\\output'

    net_type = 'small'

    data_len = 1759
    data_offset = 0

    #data_len = 272
    #data_offset = 1760

    #data_len = 1612
    #data_offset = 2032


    g_rgb, g_rgb_p = generator('rgb',model_path_rgb,model_path_depth,net_type)
    g_depth, g_depth_p = generator('depth',model_path_depth,model_path_rgb,net_type)

    H,W = 128,256
    activations_rgb_real = np.zeros((data_len, 2048), dtype=np.float32)
    activations_rgb_fake = np.zeros((data_len, 2048), dtype=np.float32)
    activations_depth_real = np.zeros((data_len, 2048), dtype=np.float32)
    activations_depth_fake = np.zeros((data_len, 2048), dtype=np.float32)

    for i in range(0,data_len):
        print(i)
        folder = 'pano_'+str(i)
        folder_offset = 'pano_'+str(i+data_offset)
        real_rgb = cv2.imread(os.path.join(gt_dir_rgb,folder_offset,folder_offset+'.jpg'))
        fake_rgb = cv2.imread(os.path.join(outdir_rgb,folder,'pano_out.jpg'))
        real_depth = cv2.imread(os.path.join(gt_dir_depth,folder_offset,folder_offset+'.jpg'))
        fake_depth = cv2.imread(os.path.join(outdir_depth,folder,'pano_out.jpg'))
        #print(np.shape(gt)) #(512,1024,3)

        real_rgb = cv2.resize(real_rgb,dsize=(W,H))
        fake_rgb = cv2.resize(fake_rgb,dsize=(W,H))
        real_depth = cv2.resize(real_depth,dsize=(W,H))
        fake_depth = cv2.resize(fake_depth,dsize=(W,H))
        #print(np.shape(f_gt)) # (1024,8,16)

        real_rgb = preprocessing(real_rgb)
        fake_rgb = preprocessing(fake_rgb)
        real_depth = preprocessing(real_depth)
        fake_depth = preprocessing(fake_depth)
        
        #real feature extraction
        p_features_rgb_real,_ = g_rgb_p(real_depth)
        _,f_rgb_real = g_rgb(real_rgb, p_features_rgb_real)
        p_features_depth_real,_ = g_depth_p(real_rgb)
        _,f_depth_real = g_depth(real_depth, p_features_depth_real)

        #fake feature extraction
        p_features_rgb_fake,_ = g_rgb_p(fake_depth)
        _,f_rgb_fake = g_rgb(fake_rgb, p_features_rgb_fake)
        p_features_depth_fake,_ = g_depth_p(fake_rgb)
        _,f_depth_fake = g_depth(fake_depth, p_features_depth_fake)
      
        
        # latitudinal_equivariance feature
        f_rgb_real = latitudinal_equivariance(f_rgb_real)
        f_rgb_fake = latitudinal_equivariance(f_rgb_fake)
        f_depth_real = latitudinal_equivariance(f_depth_real)
        f_depth_fake = latitudinal_equivariance(f_depth_fake)
   
        activations_rgb_real[i,:]=f_rgb_real[:]
        activations_rgb_fake[i,:]=f_rgb_fake[:]
        activations_depth_real[i,:]=f_depth_real[:]
        activations_depth_fake[i,:]=f_depth_fake[:]
        
    
    #np.save('matterport3d_rgb.npy',activations_rgb_real)
    #np.save('matterport3d_depth.npy',activations_depth_real)
    
    #activations_rgb_real = np.load('matterport3d_rgb.npy')
    #activations_depth_real = np.load('matterport3d_depth.npy')
    
    #print(np.shape(activations_real)) #(len,1024)
    mu1_rgb, sigma1_rgb = calc_statistics(activations_rgb_real)
    mu2_rgb, sigma2_rgb = calc_statistics(activations_rgb_fake)
    mu1_depth, sigma1_depth = calc_statistics(activations_depth_real)
    mu2_depth, sigma2_depth = calc_statistics(activations_depth_fake)
    faed_score_rgb = calc_frechet_distance(mu1_rgb,sigma1_rgb,mu2_rgb,sigma2_rgb)
    faed_score_depth = calc_frechet_distance(mu1_depth,sigma1_depth,mu2_depth,sigma2_depth)
    print('RGB: ',faed_score_rgb)
    print('DEPTH: ',faed_score_depth)

if __name__ == '__main__':
    main()