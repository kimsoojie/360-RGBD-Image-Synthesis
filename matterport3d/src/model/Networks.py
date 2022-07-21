from util.base import *
from model.models import *
from model.vgg import Vgg
import model.models as m
import model.ops as ops
from util.opt import  Options
import util.utilities as utl
from tensorboardX import SummaryWriter
from val import valdiate
import faed

opt = Options()


class Networks():
    def __init__(self, device, net_type, sobel=False, log=True, loss='gan', type='rgb'):
        print('Initiating network......', end='')
        # Networks model
        # self.Discriminator = Discriminator().to(device)
        #self.Discriminator = MultiDiscriminator().to(device)
        #self.Generator = Generator().to(device)
        if net_type == 'small':
            self.prefix = 'small'
            self.Discriminator = m.D1(6).to(device)
            self.Generator = m.GS().to(device)
            # load initial weights
            self.Discriminator.apply(init_weights)
            self.Generator.apply(init_weights)
        elif net_type == 'medium':
            self.prefix = 'medium'
            self.Discriminator = m.D1(6, loss=loss).to(device)
            self.Generator = m.GM().to(device)
            # self.Generator.decompose_layer()
            self.Discriminator.apply(init_weights)
            self.Generator.apply(init_weights)
            # load pretrained small weights
            # self.load_model_G('small_full_500000', strict=False)
        elif net_type == 'large':
            self.prefix = 'large'
            self.Discriminator = m.D1(6, loss=loss).to(device)
            self.Generator = m.GL2().to(device)
            # self.Generator = m2.GeneratorLarge().to(device)
            self.Discriminator.apply(init_weights)
            self.Generator.apply(init_weights)
            # load pretrained medium weights
            # self.load_model('medium_full_0', strict=False)

        # Initialize  Network weights
        #self.Vgg = Vgg().to(device)

        if sobel == True:
            self.use_sobel = True
            self.Sobel = Sobel().to(device)
        else:
            self.use_sobel = False

        # Network data
        self.device = device
        self.in_img = None
        self.gt_img = None
        self.in_img2 = None
        self.gt_img2 = None
        self.mask = None
        self.mask2 = None
      

        # Optimizer
        self.optim_g = optim.Adam(self.Generator.parameters(),
                                    lr=opt.learn_rate,
                                    betas=(opt.beta1, opt.beta2))
        self.optim_d = optim.Adam(self.Discriminator.parameters(),
                                lr=opt.learn_rate,
                                betas=(opt.beta1, opt.beta2))

        #if type == 'rgb':
        #    self.optim_g = optim.Adam(self.Generator.parameters(),
        #                            lr=opt.learn_rate,
        #                            betas=(opt.beta1, opt.beta2))
        #    self.optim_d = optim.Adam(self.Discriminator.parameters(),
        #                            lr=opt.learn_rate,
        #                            betas=(opt.beta1, opt.beta2))
        
        #if type == 'depth':
        #    self.optim_g = optim.Adam(self.Generator.parameters(),
        #                            lr=opt.lr_g,
        #                            betas=(opt.beta1, opt.beta2))
        #    self.optim_d = optim.Adam(self.Discriminator.parameters(),
        #                            lr=opt.lr_d,
        #                            betas=(opt.beta1, opt.beta2))

        # Network parameters
        self.phase = 'train'
        self.restore = False

        # Summary and Optimizer
        if log == True:
            self.writer = SummaryWriter(opt.train_log)
            self.writer_depth = SummaryWriter(opt.train_log_depth)
        # self.writer.add_graph(m3.GeneratorMedBalance(), (torch.zeros(1,3,256,512)), True)
        # self.writer.add_graph(m3.D1(3), (torch.zeros(1,3,256,512)), True)

        if loss == 'gan':
            self.loss_fn_GAN = nn.BCELoss()
        elif loss == 'lsgan':
            self.loss_fn_GAN = nn.MSELoss()
        # self.loss_fn_BCE = nn.BCELoss()
        # self.loss_fn_MSE = nn.MSELoss()
        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_feat = nn.L1Loss()
        self.loss_vgg = VGGLoss()
        self.faed_score = 1.0
        self.faed_max = 1.0

        # Network output
        self.d_out_fake = None
        self.d_out_real = None
        self.d_out_adv = None
        self.g_out = None
        self.g_out_sobel = None
        self.gt_sobel = None

        print('completed')

    """
        Network Forward 
        ===============
    """
    def train_small(self, in_img, gt_img):
        d_portion_loss = 0.0
        g_portion_loss = 0.0
        mask_size = [16, 32, 64, 128]
        for i in range(4):
            m = utl.create_mask_portion([mask_size[i], 256])
            self.load_input_batch(in_img * m, gt_img * m)
            self.forward_small()
            self.compute_loss_small()
            d_portion_loss += self.loss_d_total
            g_portion_loss += self.loss_g_total

        d_portion_loss.backward()
        g_portion_loss.backward()

        self.optim_d.step()
        self.optim_g.step()


    def load_input_batch(self, in_img, gt_img, fov, mask, in_img2, gt_img2, fov2, mask2):
        self.in_img = in_img
        self.gt_img = gt_img

        self.in_img2 = in_img2
        self.gt_img2 = gt_img2

        self.mask = mask
        self.mask2 = mask2

        self.preprocess_input()

        # mask = utl.create_mask_outpaint()
        # mask = mask.to(self.device)
        # self.in_img_s = self.in_img * mask
        # self.gt_img_s = self.gt_img
        # self.in_img_m = self.in_img * mask
        # self.gt_img_m = self.gt_img
        
        #self.save_img_from_tensor('.\\in_img\\'+'a.jpg', in_img)

        # 256*128
        self.in_img_s = ops.downsample(self.in_img, 4)
        self.gt_img_s = ops.downsample(self.gt_img, 4)
        # 512*256
        self.in_img_m = ops.downsample(self.in_img, 2)
        self.gt_img_m = ops.downsample(self.gt_img, 2)

        # 256*128
        self.in_img_s2 = ops.downsample(self.in_img2, 4)
        self.gt_img_s2 = ops.downsample(self.gt_img2, 4)
        # 512*256
        self.in_img_m2 = ops.downsample(self.in_img2, 2)
        self.gt_img_m2 = ops.downsample(self.gt_img2, 2)


        self.mask_s = ops.downsample(self.mask, 4)
        self.mask_m = ops.downsample(self.mask, 2)
        self.mask2_s = ops.downsample(self.mask2, 4)
        self.mask2_m = ops.downsample(self.mask2, 2)



    # 2.
    def forward_small(self):
        # 1. D real
        self.d_out_real = self.Discriminator(ops.dstack(self.in_img_s, self.gt_img_s))
        # self.d_out_real = self.Discriminator(self.gt_img_s)
        # 2. G out
        # self.g_out_s, self.g_out_ = self.Generator(self.in_img_s)
        self.g_out_s = self.Generator(self.in_img)
        # 3. D fake (detach)
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img_s,self.g_out_s).detach())
        # self.d_out_fake = self.Discriminator(self.g_out_s.detach())
        # 4. D adv
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img_s,self.g_out_s))
        # self.d_out_adv = self.Discriminator(self.g_out_s)

    def train_Gmed(self):
        self.d_out_real = torch.empty(1)
        self.d_out_fake = torch.empty(1)
        self.d_out_adv = torch.empty(1)
        self.loss_d_real = torch.empty(1)
        self.loss_d_fake = torch.empty(1)
        self.loss_g_adv = torch.empty(1)
        self.loss_d_total = torch.empty(1)

        self.Generator.zero_grad()
        self.g_out_s, self.g_out_m = self.Generator(self.in_img_m)
        self.loss_g_pix = self.loss_fn_L1(self.g_out_m, self.gt_img_m)
        self.loss_g_total = self.loss_g_pix
        self.loss_g_total.backward()
        self.optim_g.step()

    def show_img_from_tensor(self, img_tensor):
        x = img_tensor.cpu()
        x = x.detach().numpy()
        x = np.transpose(x[0,:,:,:], (1,2,0))
        img = x * 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('a',img)
        cv2.waitKey(0)

    def save_img_from_tensor(self,im_path, img_tensor):
        x = img_tensor.cpu()
        x = ((x.detach().numpy() + 1) / 2)
        x = np.transpose(x[0,:,:,:], (1,2,0))
        img = x * 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(im_path, img.astype(np.uint8))

    
    def get_faed_score(self, generator, generator_p, type='rgb'):
        #data_len = 1759
        #data_offset = 0
        net_type ='small'

        H,W = 128,256
        activations_rgb_real = np.load(opt.faed_gt_activations_rgb)
        activations_rgb_fake = np.zeros((opt.faed_data_len, 2048), dtype=np.float32)
        activations_depth_real = np.load(opt.faed_gt_activations_depth)
        activations_depth_fake = np.zeros((opt.faed_data_len, 2048), dtype=np.float32)

        g_rgb, g_rgb_p = faed.generator('rgb',opt.pretrained_model_path_rgb,opt.pretrained_model_path_depth,net_type)
        g_depth, g_depth_p = faed.generator('depth',opt.pretrained_model_path_depth,opt.pretrained_model_path_rgb,net_type)
        #g, g_p = faed.load_model(os.path.join(opt.model_path,'checkpoint_0'),
        #                         os.path.join(opt.model_path,'checkpoint_1'),net_type)

        g = generator
        g_p = generator_p

        random_fov = faed.generate_random_fov()
        mask_ones = faed.make_mask(random_fov,'ones')
        mask_zeros = faed.make_mask(random_fov,'zeros')
        print('extract faed f..')
        for i in range(0,opt.faed_data_len):
            print('extract faed f: ', i)
            folder = 'pano_'+str(i)
            #folder_offset = 'pano_'+str(i+data_offset)

            im_rgb = cv2.imread(os.path.join(opt.test_path,folder,folder+'.jpg'))
            im_depth = cv2.imread(os.path.join(opt.test_path_depth,folder,folder+'.jpg'))
            im_rgb = cv2.resize(im_rgb,dsize=(W,H))
            im_depth = cv2.resize(im_depth,dsize=(W,H))
            
            im_rgb = faed.preprocessing(im_rgb*mask_zeros)
            im_depth = faed.preprocessing(im_depth*mask_zeros)
            m_o = faed.preprocessing(mask_ones)
            
            if type == 'rgb':
                _, f1 = g(im_rgb)
                out2, _ =g_p(im_depth, f1)
                _, feature_blocks = g_p(out2*m_o)
                out, _ = g(im_rgb, feature_blocks, type)
                fake_rgb = out
                fake_depth = out2
                
                #fake feature extraction
                p_features_rgb_fake,_ = g_rgb_p(fake_depth)
                _,f_rgb_fake = g_rgb(fake_rgb, p_features_rgb_fake)

                 # latitudinal_equivariance feature
                f_rgb_fake = faed.latitudinal_equivariance(f_rgb_fake)

                activations_rgb_fake[i,:]=f_rgb_fake[:]


            if type == 'depth':
                _, f1 = g(im_depth)
                out2, _ =g_p(im_rgb, f1)
                _, feature_blocks = g_p(out2*m_o)
                out, _ = g(im_depth, feature_blocks, type)
                fake_rgb = out2
                fake_depth = out

                #fake feature extraction
                p_features_depth_fake,_ = g_depth_p(fake_rgb)
                _,f_depth_fake = g_depth(fake_depth, p_features_depth_fake)

                # latitudinal_equivariance feature
                f_depth_fake = faed.latitudinal_equivariance(f_depth_fake)
                
                activations_depth_fake[i,:]=f_depth_fake[:]

        
        if type == 'rgb':
            mu1_rgb, sigma1_rgb = faed.calc_statistics(activations_rgb_real)
            mu2_rgb, sigma2_rgb = faed.calc_statistics(activations_rgb_fake)

            faed_score_rgb = faed.calc_frechet_distance(mu1_rgb,sigma1_rgb,mu2_rgb,sigma2_rgb)

            print(type,',FAED SCORE RGB: ',faed_score_rgb)
            return faed_score_rgb
        if type == 'depth':
            mu1_depth, sigma1_depth = faed.calc_statistics(activations_depth_real)
            mu2_depth, sigma2_depth = faed.calc_statistics(activations_depth_fake)

            faed_score_depth = faed.calc_frechet_distance(mu1_depth,sigma1_depth,mu2_depth,sigma2_depth)

            print(type,',FAED SCORE DEPTH: ',faed_score_depth)
            return faed_score_depth

        
           
    
    def train_small(self, step, epoch, type = 'rgb', pair_generator = None):
        # ----------------------
        # Train discriminator
        # ----------------------
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()
        
        _, f1 = self.Generator(self.in_img_s)
        out2, _ = pair_generator(self.in_img_s2, f1)
        _, feature_blocks = pair_generator(out2 * self.mask2_s)
        self.g_out_s, _ = self.Generator(self.in_img_s, feature_blocks, type)

        #pair_g_out_s, feature_blocks = pair_generator(self.in_img_s2)
        #self.g_out_s,_ = self.Generator(self.in_img_s, feature_blocks, type)
        
        #self.g_out_s,_ = self.Generator(self.in_img_s)
        
        self.d_out_real = self.Discriminator(ops.dstack(self.in_img_s, self.gt_img_s.detach()))
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img_s, self.g_out_s.detach()))

        # nreal = np.random.uniform(low=0.9, high=1.)
        # nfake = np.random.uniform(low=0.0, high=0.1)

        # zeros = torch.zeros_like(self.d_out_fake) + nfake # 0.0 - 0.1
        # ones = torch.zeros_like(self.d_out_real) + nreal # 0.9 - 1.
        # zeros = torch.zeros_like(self.d_out_fake[0])
        # ones = torch.ones_like(self.d_out_real[0])

        # Loss
        self.loss_d_real = 0
        self.loss_d_fake = 0
        for i in range(len(self.d_out_real)):
            self.loss_d_real += self.loss_fn_GAN(self.d_out_real[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            self.loss_d_fake += self.loss_fn_GAN(self.d_out_fake[i], torch.zeros_like(self.d_out_fake[i]))
            # self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i][-1], torch.zeros_like(self.d_out_fake[i][-1]))
        self.loss_d_total = (self.loss_d_real + self.loss_d_fake)

        # Optimize
        self.loss_d_total.backward()
        self.optim_d.step()

        # ----------------------
        # Train Generator
        # ----------------------
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img_s, self.g_out_s))

        # Loss
        faed_loss = self.faed_score/(self.faed_max)
        if step%2000 == 0:
            self.faed_score = self.get_faed_score(self.Generator, pair_generator, type)
            if epoch == 0 and step == 0:
                self.faed_max = self.faed_score
            faed_loss = self.faed_score/(self.faed_max)
            print(type,',FAED LOSS: ', faed_loss)
       
        self.loss_g_adv = 0
        self.loss_g_feats = 0
        for i in range(len(self.d_out_adv)):
            self.loss_g_adv += self.loss_fn_GAN(self.d_out_adv[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i][-1}, torch.ones_like(self.d_out_real[i][-1))
            # for j in range(3):
            #     self.loss_g_feats += 0.5 * 10 *self.loss_fn_feat(self.d_out_adv[i][j], self.d_out_real[i][j].detach())
        self.loss_g_pix = self.loss_fn_L1(self.g_out_s, self.gt_img_s)
        self.loss_g_vgg = self.loss_vgg(self.g_out_s, self.gt_img_s)
        self.loss_g_total = 1 * self.loss_g_adv + 1 * self.loss_g_pix + 10 * self.loss_g_vgg + faed_loss
        # self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg + self.loss_g_feats

        #soojie: tensorboards
        
        if type == 'depth':
            self.writer_depth.add_scalars('FAED (DEPTH)', {'Score': self.faed_score}, step)
            self.writer_depth.add_scalars('GAN Loss (DEPTH)', {'Dreal': self.loss_d_real,
                                                'Dfake': self.loss_d_fake,
                                                'Dadv': self.loss_g_adv}, step)
            
            self.writer_depth.add_scalars('Total Loss (DEPTH)', {'g_pix':  self.loss_g_pix,
                                                'g_vgg': self.loss_g_vgg,
                                                'g_adv': self.loss_g_adv,
                                                'g_faed': faed_loss,
                                                'g_total': self.loss_g_total}, step)
        
        

        if type == 'rgb':
            self.writer.add_scalars('FAED', {'Score': self.faed_score}, step)
            self.writer.add_scalars('GAN Loss', {'Dreal': self.loss_d_real,
                                                'Dfake': self.loss_d_fake,
                                                'Dadv': self.loss_g_adv}, step)
            
            self.writer.add_scalars('Total Loss', {'g_pix':  self.loss_g_pix,
                                                'g_vgg': self.loss_g_vgg,
                                                'g_adv': self.loss_g_adv,
                                                'g_faed': faed_loss,
                                                'g_total': self.loss_g_total}, step)

            '''
            if(step % 1000 == 0):
                train_psnr, train_ssim, train_mse = valdiate(self.Generator,'small', False)
                test_psnr, test_ssim, test_mse = valdiate(self.Generator,'small',True)

                self.writer.add_scalars('Eval (RGB)', {'train_psnr':  train_psnr,
                                                'train_ssim': train_ssim,
                                                'train_mse': train_mse,
                                                'test_psnr':  test_psnr,
                                                'test_ssim': test_ssim,
                                                'test_mse': test_mse
                                                }, step)
            '''
         
        
        # Optimize
        self.loss_g_total.backward()
        self.optim_g.step()

        #soojie
        #if step == 0:
        #    self.save_img_from_tensor('.\\generator\\'+str(epoch)+'_in.jpg', self.in_img_s)
        #    self.save_img_from_tensor('.\\generator\\'+str(epoch)+'_out.jpg', self.g_out_s)


    def train_medium(self, step, epoch, type = 'rgb', pair_generator = None):
        # ----------------------
        # Train discriminator
        # ----------------------
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

        _, _,f1 = self.Generator(self.in_img_m)
        _,out2, _ = pair_generator(self.in_img_m2, f1)
        _, _, feature_blocks = pair_generator(out2 * self.mask2_m)
        _,self.g_out_m, _ = self.Generator(self.in_img_m, feature_blocks, type)

        self.d_out_real = self.Discriminator(ops.dstack(self.in_img_m, self.gt_img_m).detach())
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img_m, self.g_out_m.detach()))

        # nreal = np.random.uniform(low=0.9, high=1.)
        # nfake = np.random.uniform(low=0.0, high=0.1)

        # zeros = torch.zeros_like(self.d_out_fake) + nfake # 0.0 - 0.1
        # ones = torch.zeros_like(self.d_out_real) + nreal # 0.9 - 1.
        # zeros = torch.zeros_like(self.d_out_fake[0])
        # ones = torch.ones_like(self.d_out_real[0])


        # Loss
        self.loss_d_real = 0
        self.loss_d_fake = 0
        for i in range(len(self.d_out_real)):
            self.loss_d_real += self.loss_fn_GAN(self.d_out_real[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            self.loss_d_fake += self.loss_fn_GAN(self.d_out_fake[i], torch.zeros_like(self.d_out_fake[i]))
            # self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i][-1], torch.zeros_like(self.d_out_fake[i][-1]))
        self.loss_d_total = self.loss_d_real + self.loss_d_fake

        # Optimize
        self.loss_d_total.backward()
        self.optim_d.step()

        # ----------------------
        # Train Generator
        # ----------------------
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img_m, self.g_out_m))

        # Loss
        self.loss_g_adv = 0
        self.loss_g_feats = 0
        for i in range(len(self.d_out_adv)):
            self.loss_g_adv += self.loss_fn_GAN(self.d_out_adv[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i][-1}, torch.ones_like(self.d_out_real[i][-1))
            # for j in range(3):
            #     self.loss_g_feats += 0.5 * 10 *self.loss_fn_feat(self.d_out_adv[i][j], self.d_out_real[i][j].detach())
        self.loss_g_pix = self.loss_fn_L1(self.g_out_m, self.gt_img_m)
        self.loss_g_vgg = self.loss_vgg(self.g_out_m, self.gt_img_m)
        self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg
        # self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg + self.loss_g_feats

         
        #soojie: tensorboards
        if type == 'depth':
            self.writer_depth.add_scalars('GAN Loss (RGBD)', {'Dreal': self.loss_d_real,
                                                'Dfake': self.loss_d_fake,
                                                'Dadv': self.loss_g_adv}, step)
            
            self.writer_depth.add_scalars('Total Loss (RGBD)', {'g_pix':  self.loss_g_pix,
                                                'g_vgg': self.loss_g_vgg,
                                                'g_adv': self.loss_g_adv,
                                                'g_total': self.loss_g_total}, step)
            '''
            if(step % 1000 == 0):
                train_psnr_depth, train_ssim_depth, train_mse_depth = valdiate(self.Generator,'medium', False)
                test_psnr_depth, test_ssim_depth, test_mse_depth = valdiate(self.Generator,'medium',True)
                
                self.writer.add_scalars('Eval (RGBD)', {
                                                'train_psnr_depth':  train_psnr_depth,
                                                'train_ssim_depth': train_ssim_depth,
                                                'train_mse_depth': train_mse_depth,
                                                'test_psnr_depth':  test_psnr_depth,
                                                'test_ssim_depth': test_ssim_depth,
                                                'test_mse_depth': test_mse_depth
                                                }, step)
            '''

        if type == 'rgb':
            self.writer.add_scalars('GAN Loss (RGB)', {'Dreal': self.loss_d_real,
                                                'Dfake': self.loss_d_fake,
                                                'Dadv': self.loss_g_adv}, step)
            
            self.writer.add_scalars('Total Loss (RGB)', {'g_pix':  self.loss_g_pix,
                                                'g_vgg': self.loss_g_vgg,
                                                'g_adv': self.loss_g_adv,
                                                'g_total': self.loss_g_total}, step)

            '''
            if(step % 1000 == 0):
                train_psnr, train_ssim, train_mse = valdiate(self.Generator,'medium', False)
                test_psnr, test_ssim, test_mse = valdiate(self.Generator,'medium',True)

                self.writer.add_scalars('Eval (RGB)', {'train_psnr':  train_psnr,
                                                'train_ssim': train_ssim,
                                                'train_mse': train_mse,
                                                'test_psnr':  test_psnr,
                                                'test_ssim': test_ssim,
                                                'test_mse': test_mse
                                                }, step)
            '''

        # Optimize
        self.loss_g_total.backward()
        self.optim_g.step()



    def forward_medium(self):
        # 1. D real
        # rs = ops.dstack(self.in_img_s, self.gt_img_s)
        rm = ops.dstack(self.in_img_m, self.gt_img_m)
        self.d_out_real = self.Discriminator(self.gt_img_m)

        # 2. G out
        self.g_out_s, self.g_out_m = self.Generator(self.in_img)

        # 3. D fake (detach)
        # fs = ops.dstack(self.in_img_s, ops.downsample(self.g_out_m, 2))
        fm = ops.dstack(self.in_img_m, self.g_out_m)
        self.d_out_fake = self.Discriminator(self.g_out_m.detach())

        # 4. D adv
        self.d_out_adv = self.Discriminator(self.g_out_m)

    def forward_large(self):
        # 1. D real
        self.d_out_real = self.Discriminator(ops.dstack(self.in_img, self.gt_img))
        # 2. G out
        self.g_out_s, self.g_out_m, self.g_out_l = self.Generator(self.in_img)
        # 3. D fake (detach)
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img, self.g_out_l).detach())
        # 4. D adv
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img, self.g_out_l))

    def train_large(self,step,epoch, type = 'rgb', fusion=True):
        # ----------------------
        # Train discriminator
        # ----------------------
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

        self.g_out_s, self.g_out_m, self.g_out_l = self.Generator(self.in_img)

        self.d_out_real = self.Discriminator(ops.dstack(self.in_img, self.gt_img))
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img, self.g_out_l.detach()))

        # nreal = np.random.uniform(low=0.9, high=1.)
        # nfake = np.random.uniform(low=0.0, high=0.1)

        # zeros = torch.zeros_like(self.d_out_fake) + nfake # 0.0 - 0.1
        # ones = torch.zeros_like(self.d_out_real) + nreal # 0.9 - 1.
        # zeros = torch.zeros_like(self.d_out_fake[0])
        # ones = torch.ones_like(self.d_out_real[0])

        # Loss
        self.loss_d_real = 0
        self.loss_d_fake = 0
        for i in range(len(self.d_out_real)):
            self.loss_d_real += self.loss_fn_GAN(self.d_out_real[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            self.loss_d_fake += self.loss_fn_GAN(self.d_out_fake[i], torch.zeros_like(self.d_out_fake[i]))
            # self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i][-1], torch.zeros_like(self.d_out_fake[i][-1]))
        self.loss_d_total = (self.loss_d_real + self.loss_d_fake)

        # Optimize
        self.loss_d_total.backward()
        self.optim_d.step()

        # ----------------------
        # Train Generator
        # ----------------------
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img, self.g_out_l))

        # Loss
        self.loss_g_adv = 0
        self.loss_g_feats = 0
        for i in range(len(self.d_out_adv)):
            self.loss_g_adv += self.loss_fn_GAN(self.d_out_adv[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            # for j in range(3):
            #     self.loss_g_feats += 0.5 * 10 *self.loss_fn_feat(self.d_out_adv[i][j], self.d_out_real[i][j].detach())
        self.loss_g_pix = self.loss_fn_L1(self.g_out_l, self.gt_img)
        self.loss_g_vgg = self.loss_vgg(self.g_out_l, self.gt_img)
        self.loss_g_total = 1 * self.loss_g_adv + 1 * self.loss_g_pix + 10 * self.loss_g_vgg
        # self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg + self.loss_g_feats

        #soojie: tensorboards
        if type == 'depth':
            self.writer_depth.add_scalars('GAN Loss (RGBD)', {'Dreal': self.loss_d_real,
                                                'Dfake': self.loss_d_fake,
                                                'Dadv': self.loss_g_adv}, step)
            
            self.writer_depth.add_scalars('Total Loss (RGBD)', {'g_pix':  self.loss_g_pix,
                                                'g_vgg': self.loss_g_vgg,
                                                'g_adv': self.loss_g_adv,
                                                'g_total': self.loss_g_total}, step)
            
            if(step % 1000 == 0):
                train_psnr_depth, train_ssim_depth, train_mse_depth = valdiate(self.Generator,'large', False)
                test_psnr_depth, test_ssim_depth, test_mse_depth = valdiate(self.Generator,'large',True)
                
                self.writer.add_scalars('Eval (RGBD)', {
                                                'train_psnr_depth':  train_psnr_depth,
                                                'train_ssim_depth': train_ssim_depth,
                                                'train_mse_depth': train_mse_depth,
                                                'test_psnr_depth':  test_psnr_depth,
                                                'test_ssim_depth': test_ssim_depth,
                                                'test_mse_depth': test_mse_depth
                                                }, step)
            

        if type == 'rgb':
            self.writer.add_scalars('GAN Loss (RGB)', {'Dreal': self.loss_d_real,
                                                'Dfake': self.loss_d_fake,
                                                'Dadv': self.loss_g_adv}, step)
            
            self.writer.add_scalars('Total Loss (RGB)', {'g_pix':  self.loss_g_pix,
                                                'g_vgg': self.loss_g_vgg,
                                                'g_adv': self.loss_g_adv,
                                                'g_total': self.loss_g_total}, step)

            
            if(step % 1000 == 0):
                train_psnr, train_ssim, train_mse = valdiate(self.Generator,'large',True, False)
                test_psnr, test_ssim, test_mse = valdiate(self.Generator,'large', True,True)

                self.writer.add_scalars('Eval (RGB)', {'train_psnr':  train_psnr,
                                                'train_ssim': train_ssim,
                                                'train_mse': train_mse,
                                                'test_psnr':  test_psnr,
                                                'test_ssim': test_ssim,
                                                'test_mse': test_mse
                                                }, step)
        # Optimize
        self.loss_g_total.backward()
        self.optim_g.step()


    def compute_loss_small(self):
        ones = torch.ones_like(self.d_out_real)
        zeros = torch.zeros_like(self.d_out_fake)

        # 1. loss D
        self.loss_d_real = self.loss_fn_GAN(self.d_out_real, ones)
        self.loss_d_fake = self.loss_fn_GAN(self.d_out_fake, zeros)
        self.loss_d_total = 0.5 * (self.loss_d_real + self.loss_d_fake)

        # 2. loss G
        ones_ = torch.ones_like(self.d_out_adv)
        self.loss_g_adv = self.loss_fn_GAN(self.d_out_adv, ones_)
        self.loss_g_vgg = self.loss_vgg(self.g_out_s, self.gt_img_s)
        self.loss_g_pix = self.loss_fn_L1(self.g_out_s, self.gt_img_s)
        # self.loss_g_pix = self.loss_fn_L1(self.g_out_m, self.gt_img_m)
        # self.loss_g_pix = self.loss_fn_L1(self.g_out_l, self.gt_img)
        self.loss_g_total =  0.002 * self.loss_g_adv + 1 * (self.loss_g_pix) +  0.1 * self.loss_g_vgg

    # 3.
    def compute_loss_medium(self):
        ones = torch.ones_like(self.d_out_real)
        zeros = torch.zeros_like(self.d_out_fake)

        # 1. loss D
        self.loss_d_real = self.loss_fn_MSE(self.d_out_real, ones)
        self.loss_d_fake = self.loss_fn_MSE(self.d_out_fake, zeros)
        self.loss_d_total = 0.5 * (self.loss_d_real + self.loss_d_fake)

        # 2. loss G
        ones_ = torch.ones_like(self.d_out_adv)
        self.loss_g_adv = self.loss_fn_MSE(self.d_out_adv, ones_)
        # self.loss_g_pix = self.loss_fn_L1(self.g_out_m, self.gt_img_m)
        self.loss_g_vgg = self.loss_vgg(self.g_out_m, self.gt_img_m)
        # self.loss_g_total = self.loss_g_adv + 100 * (self.loss_g_pix) + self.loss_g_vgg
        # self.loss_g_total =  0.002 * self.loss_g_adv + (self.loss_g_pix) + 0.2 * self.loss_g_vgg
        self.loss_g_total =  0.001 * self.loss_g_adv + 1 * self.loss_g_vgg

    def compute_loss_large(self):
        ones = torch.ones_like(self.d_out_real)
        zeros = torch.zeros_like(self.d_out_fake)

        # 1. loss D
        self.loss_d_real = self.loss_fn_GAN(self.d_out_real, ones)
        self.loss_d_fake = self.loss_fn_GAN(self.d_out_fake, zeros)
        self.loss_d_total = 0.5 * (self.loss_d_real + self.loss_d_fake)

        # 2. loss G
        ones_ = torch.ones_like(self.d_out_adv)
        self.loss_g_adv = self.loss_fn_GAN(self.d_out_adv, ones_)
        self.loss_g_pix = self.loss_fn_L1(self.g_out_l, self.gt_img)
        self.loss_g_vgg = self.loss_vgg(self.g_out_l, self.gt_img)
        # self.loss_g_total = 0.02 * self.loss_g_adv + (self.loss_g_pix) + 0.2 * self.loss_g_vgg
        self.loss_g_total =  0.002 * self.loss_g_adv + (self.loss_g_pix) + 0.2 * self.loss_g_vgg

    # 4.
    def optimize(self):
        # 1. backprop and update D
        self.loss_d_total.backward()
        self.optim_d.step()

        # 2. backprop and update G
        self.loss_g_total.backward()
        self.optim_g.step()

    # 4.
    def optimize_D(self):
        # 1. backprop and update D
        self.loss_d_total.backward()
        self.optim_d.step()

    # 4.
    def optimize_G(self):
        # 2. backprop and update G
        self.loss_g_total.backward()
        self.optim_g.step()


    """
        Network losses
        ===============
    """
    def compute_loss_D(self):
        # self.loss_d_real = 0
        # self.loss_d_fake = 0
        # for i in range(3):
        #     ones = torch.ones_like(self.d_out_real[i])
        #     zeros = torch.zeros_like(self.d_out_fake[i])
        #     self.loss_d_real += self.loss_fn_BCE(self.d_out_real[i], ones)
        #     self.loss_d_fake += self.loss_fn_BCE(self.d_out_fake[i], zeros)
        ones = torch.ones_like(self.d_out_real)
        zeros = torch.zeros_like(self.d_out_real)
        self.loss_d_real = self.loss_fn_BCE(self.d_out_real, ones)
        self.loss_d_fake = self.loss_fn_BCE(self.d_out_fake, zeros)

        self.loss_d_total = 0.5 * (self.loss_d_real + self.loss_d_fake)
        return self.loss_d_total

    def compute_loss_vgg(self, gt_img):
        output = self.Vgg(self.g_out)
        gt = self.Vgg(gt_img)

        loss = 0
        w = [1/32.0, 1/16.0, 1.0/8, 1.0/4, 1.0]
        for i in range(2):
            loss += w[i] * self.loss_fn_L1(output[i], gt[i])
        return loss

    def compute_loss_G(self, gt_img):
        self.loss_g_sobel = 0
        self.loss_g_vgg = 0
        # Adversarial loss
        # self.loss_g_adv = 0
        # for i in range(3):
        #     ones = torch.ones_like(self.d_out_adv[i])
        #     self.loss_g_adv += self.loss_fn_BCE(self.d_out_adv[i], ones)
        ones = torch.ones_like(self.d_out_adv)
        self.loss_g_adv = self.loss_fn_BCE(self.d_out_adv, ones)

        # Sobel loss
        if self.use_sobel == True:
            self.g_out_sobel = self.Sobel(self.g_out) # Compute sobel
            self.gt_sobel = self.Sobel(gt_img.detach()) # compute sobel
            for i in range(2):
                self.loss_g_sobel += self.loss_fn_L1(self.g_out_sobel[i], self.gt_sobel[i])

        # Compute total loss
        self.loss_g_pix = self.loss_fn_L1(self.g_out, gt_img)
        #self.loss_g_vgg = self.compute_loss_vgg(gt_img)
        self.loss_g_total = self.loss_g_adv + 100 * self.loss_g_pix + self.loss_g_sobel
        return self.loss_g_total


    """
        Saving and loading Model
        ========================
    """
    def save_model(self, step, model_name='model'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step))
        torch.save({'Generator': self.Generator.state_dict(),
                    'Discriminator' : self.Discriminator.state_dict()},
                   model_path + '.pt')
        print('Finished saving model')

    def save_ckpt(self, step, model_name='checkpoint'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step))
        torch.save({'step': step,
                    'Generator': self.Generator.state_dict(),
                    'Discriminator' : self.Discriminator.state_dict()},
                   model_path + '.pt')
        return 0

    def load_model(self, model_name, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        model = torch.load(model_path)
        self.Discriminator.load_state_dict(model['Discriminator'], strict=strict)
        self.Generator.load_state_dict(model['Generator'], strict=strict)
        print('Finished loading trained model')

    def load_model_G(self, model_name, gan=True, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        if gan == True:
            model = torch.load(model_path)
            self.Generator.load_state_dict(model['Generator'], strict=strict)
        else:
            model = torch.load(model_path)
            self.Generator.load_state_dict(model, strict=strict)
        print('Finished loading trained model')

    def restore_ckpt(self):
        return 0


    """
        Training Summary
        ================
    """
    def write_imgs_summary_small(self, step):
        g_out_s =self.g_out_s[0,:,:,:]
        # g_out_ =self.g_out_[0,:,:,:]
        in_img_s =self.in_img_s[0,:,:,:]
        gt_img_s =self.gt_img_s[0,:,:,:]
        self.writer.add_image('out/output_small', (g_out_s+1)/2,  step)
        # self.writer.add_image('out/output_res', (g_out_+1)/2,  step)
        self.writer.add_image('in/input_small', (in_img_s+1)/2,  step)
        self.writer.add_image('in/gt_small', (gt_img_s+1)/2,  step)

    def write_scalars(self, step):
        self.writer.add_scalars('GAN Loss', {'Dreal': self.loss_d_real,
                                             'Dfake': self.loss_d_fake,
                                             'Dadv': self.loss_g_adv}, step)

    def write_imgs_summary_medium(self, step):
        # Small
        # Medium
        g_out_m = self.g_out_m[0,:,:,:]
        # g_out_s = torch.tanh(self.g_out_s[0,:,:,:])
        in_img_m = self.in_img_m[0,:,:,:]
        gt_img_m = self.gt_img_m[0,:,:,:]
        # self.writer.add_image('out/output_small', (g_out_s.squeeze(0)+1)/2,  step)
        self.writer.add_image('out/output_medium', (g_out_m.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/input_medium', (in_img_m.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/gt_medium', (gt_img_m.squeeze(0)+1)/2,  step)

    def write_imgs_summary_large(self, step):
        g_out_s = self.g_out_s[0,:,:,:]
        g_out_m = self.g_out_m[0,:,:,:]
        g_out_l = self.g_out_l[0,:,:,:]
        in_img = self.in_img[0,:,:,:]
        gt_img = self.gt_img[0,:,:,:]
        # Small
        self.writer.add_image('out/output_small', (g_out_s.squeeze(0)+1)/2,  step)
        # Medium
        self.writer.add_image('out/output_medium', (g_out_m.squeeze(0)+1)/2,  step)
        # Large
        self.writer.add_image('out/output_large', (g_out_l.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/input_large', (in_img.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/gt_large', (gt_img.squeeze(0)+1)/2,  step)

    # def write_imgs_summary(self, input, gt, step):
    #     output = self.g_out
    #     self.writer.add_image('output', (output.squeeze(0)+1)/2,  step)
    #     self.writer.add_image('input', (input.squeeze(0)+1)/2,  step)
    #     self.writer.add_image('gt', (gt.squeeze(0)+1)/2,  step)

    def write_img_summary(self, input, name, step):
        self.writer.add_image(name, (input.squeeze(0)+1)/2, step)

    def write_scalar_summary(self, tag, step):
        self.writer.add_scalar(tag, self.loss_g_pix, step)

    def print_summary_small(self, epoch, step, type = 'rgb'):
        real = 0
        fake = 0
        # for i in range(3):
        #     # compute mean of pixel
        #     real += torch.mean(self.d_out_real[i])
        #     fake += torch.mean(self.d_out_adv[i])
        # # Mean of all resolution
        # real = real / 3.0
        # fake = fake / 3.0
        real = torch.mean(self.d_out_real[0][-1])
        fake = torch.mean(self.d_out_adv[0][-1])
        if type == 'rgb':
            print('RGB: Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))
        if type == 'depth':
            print('DEPTH: Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len_depth, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))

    def print_summary_medium(self, epoch, step, type = 'rgb'):
        real = 0
        fake = 0
        for i in range(2):
            # compute mean of pixel
            real += torch.mean(self.d_out_real[i])
            # real += torch.mean(self.d_out_real[i][-1])
            fake += torch.mean(self.d_out_adv[i])
            # fake += torch.mean(self.d_out_adv[i][-1])
        # Mean of all resolution
        real = real / 3.0
        fake = fake / 3.0
        if type == 'rgb':
            print('RGB: Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))
        if type == 'depth':
            print('DEPTH: Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len_depth, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))

    def print_summary_large(self, epoch, step):
        return

    """
        Network related function
        ========================
    """
    def print_structure(self):
        print(self.Generator)
        print(self.Discriminator)
        print('Phase : ' + self.phase)

    def clear_gradient(self):
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

    def preprocess_input(self):
        # self.in_img = utl.cube_to_equirect(self.in_img, opt.equi_coord)
        self.gt_img = self.gt_img.to(self.device)
        self.in_img = self.in_img.to(self.device)

        self.gt_img2 = self.gt_img2.to(self.device)
        self.in_img2 = self.in_img2.to(self.device)

        self.mask = self.mask.to(self.device)
        self.mask2 = self.mask2.to(self.device)


    def set_phase(self, phase='train'):
        if phase == 'test':
            print('Network phase : Test')
            self.phase = 'test'
            self.Generator.eval()
            self.Discriminator.eval()
        else:
            print('Network phase : Train')
            self.Generator.train()
            self.Discriminator.train()


class Networks_Wnet():
    def __init__(self, device, num_class):
        self.num_class = num_class
        self.Wnet = Wnet(self.num_class).to(device)

        self.in_img = None

        self.optimizer = optim.Adam(self.Wnet.parameters(),
                                        lr=opt.learn_rate, betas=(opt.beta1, opt.beta2))

        self.writer = SummaryWriter(opt.train_log)

        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_L2 = nn.MSELoss()

        # Output
        self.out_segment = None
        self.out_recon = None
        self.net_loss = None

    # Networks operations
    # ===========================
    # 1. Load Input image
    def load_input_batch(self, in_img, gt_img=None):
        self.in_img = ops.downsample(in_img, 4) # 256x128

    # 2. Run forward ops
    def forward(self):
        # clear gradient and forward
        self.Wnet.zero_grad()
        self.out_recon, self.out_segment = self.Wnet.forward(self.in_img)

    # 3. Compute loss
    def compute_loss(self):
        self.net_loss = self.loss_fn_L2(self.out_recon, self.in_img)

    # 4. Update network
    def optimize(self):
        # Backpropagate and update
        self.net_loss.backward()
        self.optimizer.step()

    # Networks utilities
    # ===========================
    def set_phase(self, phase):
        if phase == 'test':
            self.Wnet.eval()
        else:
            self.Wnet.train()

    def print_summary(self, epoch, step):
        print('Epoch %d [%d | %d] > loss : %.5f'
              %(epoch, step, opt.train_len, self.net_loss))

    def write_img_summary(self, step):
        self.writer.add_image('Input', (self.in_img.squeeze(0)+1)/2, step)
        self.writer.add_image('Output', (self.out_recon.squeeze(0)+1)/2, step)
        segment = utl.colorize(self.out_segment, self.num_class)
        self.writer.add_image('Segmentation', (segment.squeeze(0)), step)


    # Saving and Loading model
    # ===========================
    def save_model(self, step, model_name='wnet_'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step))
        torch.save(self.Wnet.state_dict(), model_path + '.pt')
        print('Finished saving model')

    def load_model(self, model_name, strict=True):
       print('Loading trained model')
       model_path = os.path.join(opt.model_path, model_name + '.pt')
       model = torch.load(model_path)
       self.Wnet.load_state_dict(model, strict=strict)
       print('Finished loading trained model')


class NetworksSegment():
    def __init__(self, device, num_class):
        self.Generator = GeneratorSmall(3+num_class, 3).to(device)
        self.Generator_segment = Unet(3, 3+num_class).to(device)
        self.Discriminator = DiscriminatorSmall().to(device)
        self.num_class = num_class

        self.device = device
        self.Generator.apply(init_weights)
        self.Generator_segment.apply(init_weights)
        self.Discriminator.apply(init_weights)

        self.writer = SummaryWriter(opt.train_log)
        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_GAN = nn.BCELoss()

        # Optimizer
        self.optim_s = optim.Adam(self.Generator_segment.parameters(),
                                  lr=opt.learn_rate*10)
        self.optim_g = optim.Adam(self.Generator.parameters(),
                                  lr=opt.learn_rate,
                                  betas=(opt.beta1, opt.beta2))
        self.optim_d = optim.Adam(self.Discriminator.parameters(),
                                  lr=opt.learn_rate,
                                  betas=(opt.beta1, opt.beta2))



    # 1.
    def load_input_batch(self, in_img, gt_img):
        self.in_img = in_img
        self.gt_img = gt_img

        self.preprocess_input()


        self.in_img_ori = self.in_img.clone()
        self.gt_img_ori = self.gt_img.clone()

        if opt.net == 'small':
            self.in_img = ops.downsample(self.in_img, 4)
            self.gt_img = ops.downsample(self.gt_img, 4)
        elif opt.net == 'medium':
            ops.downsample(self.in_img, 2)
            ops.downsample(self.gt_img, 2)

    # 2.
    def forward(self):
        seg = self.Generator_segment(self.in_img_ori)
        self.g_pix = seg[0]
        self.g_seg = seg[1]

        # 1. D real
        self.d_out_real = self.Discriminator(ops.dstack(self.in_img, self.gt_img))
        # 2. G out
        self.g_out = self.Generator(ops.dstack(self.in_img_ori, self.g_seg).detach())
        # 3. D fake (detach)
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img, self.g_out).detach())
        # 4. D adv
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img, self.g_out))

    # 3.
    def compute_loss(self):
        ones = torch.ones_like(self.d_out_real)
        zeros = torch.zeros_like(self.d_out_fake)

        # 0. loss Seg
        self.loss_g_seg = self.loss_fn_L1(self.g_pix, self.gt_img_ori)

        # 1. loss D
        self.loss_d_real = self.loss_fn_GAN(self.d_out_real, ones)
        self.loss_d_fake = self.loss_fn_GAN(self.d_out_fake, zeros)
        self.loss_d_total = 0.5 * (self.loss_d_real + self.loss_d_fake)

        # 2. loss G
        ones_ = torch.ones_like(self.d_out_adv)
        self.loss_g_adv = self.loss_fn_GAN(self.d_out_adv, ones_)
        self.loss_g_pix = self.loss_fn_L1(self.g_out, self.gt_img)
        self.loss_g_total = self.loss_g_adv + 100 * (self.loss_g_pix)



    # 4.
    def optimize(self):
        # 0. backprop and update Gseg
        self.loss_g_seg.backward()
        self.optim_s.step()

        # 1. backprop and update D
        self.loss_d_total.backward()
        self.optim_d.step()

        # 2. backprop and update G
        self.loss_g_total.backward()
        self.optim_g.step()

    # 4.
    def optimize_D(self):
        # 1. backprop and update D
        self.loss_d_total.backward()
        self.optim_d.step()

    # 4.
    def optimize_G(self):
        # 2. backprop and update G
        self.loss_g_total.backward()
        self.optim_g.step()


    """
        Network operations
        =============================
    """
    def clear_gradient(self):
        self.Generator.zero_grad()
        self.Generator_segment.zero_grad()
        self.Discriminator.zero_grad()

    def preprocess_input(self):
        self.in_img = utl.cube_to_equirect(self.in_img, opt.equi_coord)
        self.gt_img = self.gt_img.to(self.device)
        self.in_img = self.in_img.to(self.device)

    def set_phase(self, phase='train'):
        if phase == 'test':
            self.Generator.train()
            self.Discriminator.train()
        else:
            self.Generator.eval()
            self.Discriminator.eval()


    """
        Saving and loading model
        =============================
    """
    def load_model(self, model_name, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        model = torch.load(model_path)
        self.Generator.load_state_dict(model['Generator'], strict=strict)
        self.Discriminator.load_state_dict(model['Discriminator'], strict=strict)
        print('Finished loading trained model')

    def save_model(self, step, model_name='model'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step))
        torch.save({'Generator': self.Generator.state_dict(),
                    'Segment' : self.Generator_segment.state_dict(),
                    'Discriminator' : self.Discriminator.state_dict()},
                   model_path + '.pt')
        print('Finished saving model')


    """
        Network utilities
        =============================
    """
    def print_summary(self, epoch, step):
        real = torch.mean(self.d_out_real)
        fake = torch.mean(self.d_out_adv)
        print('Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))

    def write_imgs_summary(self, step):
        self.writer.add_image('out/output', (self.g_out.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/input', (self.in_img.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/gt', (self.gt_img.squeeze(0)+1)/2,  step)
        segment = utl.colorize(self.g_seg, self.num_class)
        self.writer.add_image('out/segment', segment.squeeze(0), step)
        self.writer.add_image('out/recon', (self.g_pix.squeeze(0)+1)/2,  step)

