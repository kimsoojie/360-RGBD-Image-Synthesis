from util.base import *
from util.utilities import numpy_to_pano
import glob
from util.opt import Options

class PanoData(Dataset):
    def __init__(self, root_dir, data_len, transform=None, data_path=None):
        self.root_dir = root_dir
        self.data_len = data_len
        self.transform = transform
        
        self.offset=0
        self.count = 0+self.offset
        self._numbers = [n for n in range(0, 14510)]

        # with open(root_dir, 'r') as f:
        #     self.data_path =  f.read().splitlines()
        # self.data_len = len(self.data_path)

    def __getitem__(self, idx):
        
        #sub_dir = 'pano_' + str(idx + 1)
        if self.count == self.data_len+self.offset:
            self.count = 0+self.offset
        
        sub_dir = 'pano_' + str( self._numbers[self.count])
        self.count += 1

        """Standard"""
        # ----------
        # in_img_cat = self._read_pano(sub_dir, prefix='pre_input45.jpg', scale=1.0)
        
        in_img_cat, fov = self._read_rand_img_pano(sub_dir, prefix='gt_') # generate random fov image
        gt_img_cat = self._read_pano(sub_dir, prefix='pano_*.jpg', scale=1.0) # pano groundtruth
        # in_img_cat = self._concat_img(sub_dir, prefix='new_img_')
        # in_img_cat = gt_img_cat
        # fov = self._read_fov(sub_dir, prefix='fov.txt')

        #cv2.imshow('img_in', in_img_cat)
        #cv2.imshow('img_gt', gt_img_cat)
        #cv2.waitKey(0)

        """Random for FOV"""
        # --------------
        # in_img_cat, gt_img_cat, fov = self._read_rand_img(sub_dir, prefix='gt_') # generate random fov image
        # cv2.imshow('img', gt_img_cat)
        # cv2.waitKey(0)

        """Read Data from list"""
        # in_img_cat = self._imread(self.data_path[idx])
        # #in_img_cat = self._read_pano(sub_dir, prefix='pano_*.jpg.jpg', scale=1.0)
        # gt_img_cat = in_img_cat
        # fov = 0

        #mask = self._make_mask(fov)
        mask = self._make_mask(212)  #90-degree

        in_img_cat = in_img_cat/127.5 - 1
        gt_img_cat = gt_img_cat/127.5 - 1

        sample = {'input': in_img_cat, 'gt': gt_img_cat, 'fov': fov, 'dir': sub_dir, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.data_len

    def _concat_img_full(self, sub_dir, prefix='gt_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        empty = np.zeros_like(images[0])
        img_top = np.hstack((empty, images[4], empty, empty))
        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_bot = np.hstack((empty, images[5], empty, empty))

        img_concat_full = np.vstack((img_top, img_concat, img_bot))
        return img_concat_full

    def _concat_img(self, sub_dir, prefix='new_img_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))

        return img_concat

    def _concat_img_pad(self, sub_dir, prefix='im_'):
        images = []
        for i in range(4):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread_pad(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_concat_full = np.vstack((np.zeros_like(img_concat), img_concat, np.zeros_like(img_concat)))
        return img_concat_full

    def _read_fov(self, sub_dir, prefix='fov.txt'):
        out = np.zeros((1,128))
        file_path = os.path.join(self.root_dir, sub_dir, prefix)
        with open(file_path) as f:
            for line in f:
                idx = line.strip()
                idx = int(int(idx)/2)
                out[0, idx-1] = 1
        return idx

    def _read_rand_img(self, sub_dir, prefix='gt_'):
        images = []
        gts = []
        fov = self.generate_random_fov()

        for i in range(4):
            img_path = os.path.join(self.root_dir, sub_dir, prefix + str(i+1) + '.jpg')
            im = self._imread(img_path)
            images.append(self.generate_crop_img(im, fov))
            gts.append(self.generate_pad_img(im, fov, pad_type='zeros'))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        gt_concat = np.hstack((gts[2], gts[0], gts[3], gts[1]))
        fov = int(fov/2) # downsample image twice
        return img_concat, gt_concat, fov

    def _make_mask(self, fov, scale=1.0):
        empty = np.zeros((256,256,3), np.uint8)
        f = self.generate_pad_img(empty,fov,pad_type='ones')
        empty_horiz = np.hstack([f,f,f,f])
        empty_cat = np.vstack([np.ones_like(empty_horiz), empty_horiz, np.ones_like(empty_horiz)])
        empty_cat = numpy_to_pano(empty_cat)
        #cv2.imwrite('./mask.jpg', empty_cat)
        #img = cv2.imread('.\\mask.jpg')
        img_rsz = cv2.resize(empty_cat, (0,0), fx=scale, fy=scale)
        return img_rsz
    '''
    def numpy_to_pano(self, in_img, out_h=512, out_w=1024, in_len=256):
        out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        coord = np.load('util/pano_coord_1024.npy')
    
        y = coord[:, :, 0]  # load the coordinate
        x = coord[:, :, 1]  # load the coordinate
        out_img = in_img[y, x, :]
        return out_img
    '''
    def _read_rand_img_pano(self, sub_dir, prefix='gt_'):
        images = []
        gts = []
        fov = self.generate_random_fov()
        #fov = self.generate_fov_order(self.count, self.offset)

        for i in range(4):
            img_path = os.path.join(self.root_dir, sub_dir, prefix + str(i+1) + '.jpg')
            #print(img_path)
            im = self._imread(img_path)
            gts.append(self.generate_pad_img(im, fov, pad_type='zeros'))

        gt_concat = np.hstack((gts[2], gts[0], gts[3], gts[1]))
        gt_concat_full = np.vstack((np.zeros_like(gt_concat), gt_concat, np.zeros_like(gt_concat)))
        
        pano = numpy_to_pano(gt_concat_full)
        fov = int(fov)
        return pano, fov

    def _read_pano(self, sub_dir, prefix='pano_', scale=1.0):
        im_path_ = os.path.join(self.root_dir, sub_dir, prefix)
        im_list = glob.glob(im_path_)
        img = self._imread(im_list[0])
        img_rsz = cv2.resize(img, (0,0), fx=scale, fy=scale)
        return img_rsz

    def _imread(self, x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _imread_rsz(self,x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        return img

    def _imread_pad(self,x):
        img = self._imread(x)
        img = cv2.resize(img, (128, 128))
        pad = 64
        img = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_CONSTANT)
        return img

    def _edge_img(self, x, y):
        return

    def generate_fov_order(self, idx, offset):
        # idx 0~64 value
        order = idx - offset
        order = order - 64*int(order/64)
        if order%2 != 0:
            order = order + 1

        # return 128~192 even number
        return int(192-order)

    def generate_random_fov(self):
        fov_range = np.arange(128,212,2)
        np.random.shuffle(fov_range)
        return fov_range[0]

    def generate_crop_img(self, img, fov):
        pad = int((256-fov)/2)
        x = img[pad:pad+fov,pad:pad+fov,:]
        x = cv2.resize(x,(256,256))
        return x

    def generate_pad_img(self, img, fov, pad_type = 'zeros'):
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


class PanoData_val(Dataset):
    def __init__(self, root_dir, data_len, transform=None, data_path=None):
        self.opt = Options(sys.argv[0])
        self.root_dir = root_dir
        self.data_len = data_len -1
        self.transform = transform

        # with open(root_dir, 'r') as f:
        #     self.data_path =  f.read().splitlines()
        # self.data_len = len(self.data_path)

    def __getitem__(self, idx):
        #1~359
        sub_dir = 'pano_' + str(idx + 1 + self.opt.train_len)

        """Standard"""
        # ----------
        # in_img_cat = self._read_pano(sub_dir, prefix='pre_input45.jpg', scale=1.0)
        in_img_cat, fov = self._read_rand_img_pano(sub_dir, prefix='gt_') # generate random fov image
        gt_img_cat = self._read_pano(sub_dir, prefix='pano_*.jpg', scale=1.0) # pano groundtruth
        # in_img_cat = self._concat_img(sub_dir, prefix='new_img_')
        # in_img_cat = gt_img_cat
        # fov = self._read_fov(sub_dir, prefix='fov.txt')

        """Random for FOV"""
        # --------------
        # in_img_cat, gt_img_cat, fov = self._read_rand_img(sub_dir, prefix='gt_') # generate random fov image
        # cv2.imshow('img', gt_img_cat)
        # cv2.waitKey(0)

        """Read Data from list"""
        # in_img_cat = self._imread(self.data_path[idx])
        # #in_img_cat = self._read_pano(sub_dir, prefix='pano_*.jpg.jpg', scale=1.0)
        # gt_img_cat = in_img_cat
        # fov = 0

        in_img_cat = in_img_cat/127.5 - 1
        gt_img_cat = gt_img_cat/127.5 - 1

        sample = {'input': in_img_cat, 'gt': gt_img_cat, 'fov': fov, 'dir': sub_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.data_len

    def _concat_img_full(self, sub_dir, prefix='gt_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        empty = np.zeros_like(images[0])
        img_top = np.hstack((empty, images[4], empty, empty))
        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_bot = np.hstack((empty, images[5], empty, empty))

        img_concat_full = np.vstack((img_top, img_concat, img_bot))
        return img_concat_full

    def _concat_img(self, sub_dir, prefix='new_img_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))

        return img_concat

    def _concat_img_pad(self, sub_dir, prefix='im_'):
        images = []
        for i in range(4):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread_pad(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_concat_full = np.vstack((np.zeros_like(img_concat), img_concat, np.zeros_like(img_concat)))
        return img_concat_full

    def _read_fov(self, sub_dir, prefix='fov.txt'):
        out = np.zeros((1,128))
        file_path = os.path.join(self.root_dir, sub_dir, prefix)
        with open(file_path) as f:
            for line in f:
                idx = line.strip()
                idx = int(int(idx)/2)
                out[0, idx-1] = 1
        return idx

    def _read_rand_img(self, sub_dir, prefix='gt_'):
        images = []
        gts = []
        fov = self.generate_random_fov()

        for i in range(4):
            img_path = os.path.join(self.root_dir, sub_dir, prefix + str(i+1) + '.jpg')
            im = self._imread(img_path)
            images.append(self.generate_crop_img(im, fov))
            gts.append(self.generate_pad_img(im, fov))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        gt_concat = np.hstack((gts[2], gts[0], gts[3], gts[1]))
        fov = int(fov/2) # downsample image twice
        return img_concat, gt_concat, fov

    def _read_rand_img_pano(self, sub_dir, prefix='gt_'):
        images = []
        gts = []
        fov = self.generate_random_fov()

        for i in range(4):
            img_path = os.path.join(self.root_dir, sub_dir, prefix + str(i+1) + '.jpg')
            #print(img_path)
            im = self._imread(img_path)
            gts.append(self.generate_pad_img(im, fov))

        gt_concat = np.hstack((gts[2], gts[0], gts[3], gts[1]))
        gt_concat_full = np.vstack((np.zeros_like(gt_concat), gt_concat, np.zeros_like(gt_concat)))
        pano = numpy_to_pano(gt_concat_full)
        fov = int(fov)
        return pano, fov

    def _read_pano(self, sub_dir, prefix='pano_', scale=1.0):
        im_path_ = os.path.join(self.root_dir, sub_dir, prefix)
        im_list = glob.glob(im_path_)
        img = self._imread(im_list[0])
        img_rsz = cv2.resize(img, (0,0), fx=scale, fy=scale)
        return img_rsz

    def _imread(self, x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _imread_rsz(self,x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        return img

    def _imread_pad(self,x):
        img = self._imread(x)
        img = cv2.resize(img, (128, 128))
        pad = 64
        img = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_CONSTANT)
        return img

    def _edge_img(self, x, y):
        return

    def generate_random_fov(self):
        fov_range = np.arange(128,192,2)
        np.random.shuffle(fov_range)
        return fov_range[0]

    def generate_crop_img(self, img, fov):
        pad = int((256-fov)/2)
        x = img[pad:pad+fov,pad:pad+fov,:]
        x = cv2.resize(x,(256,256))
        return x

    def generate_pad_img(self, img, fov):
        pad = int((256-fov)/2)

        x = int((256-fov)/2)
        y = int((256-fov)/2)
        w = fov
        h = fov
        img = cv2.resize(img, (256,256))
        img = img[y:y+h,x:x+w]

        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        return img