# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 08:43:22 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""
import glob,os

def imgs_layoutShow_FPList(imgs_fp_list,columns,scale,figsize=(15,10)):
    import math,os
    import matplotlib.pyplot as plt
    from PIL import Image
    '''
    function - 显示一个文件夹下所有图片，便于查看。

    Paras:
    imgs_root - 图像所在根目录
    imgsFn_lst - 图像名列表
    columns - 列数
    '''
    rows=math.ceil(len(imgs_fp_list)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=figsize,)   #布局多个子图，每个子图显示一幅图像 sharex=True,sharey=True,
    ax=axes.flatten()  #降至1维，便于循环操作子图
    for i in range(len(imgs_fp_list)):
        img_path=imgs_fp_list[i] #获取图像的路径
        img_array=Image.open(img_path) #读取图像为数组，值为RGB格式0-255        
        img_resize=img_array.resize([int(scale * s) for s in img_array.size] ) #传入图像的数组，调整图片大小
        ax[i].imshow(img_resize,)  #显示图像 aspect='auto'
        ax[i].set_title(i+1)
    invisible_num=rows*columns-len(imgs_fp_list)
    if invisible_num>0:
        for i in range(invisible_num):
            ax.flat[-(i+1)].set_visible(False)
    fig.tight_layout() #自动调整子图参数，使之填充整个图像区域  
    fig.suptitle("images show",fontsize=14,fontweight='bold',y=1.02)
    plt.show()

class combine_pics:
    def __init__(self,save_path,file_path,n_cols,scale,space=1,pad_val=255,figsize=(20,10)):
        import os,math
        self.save_path=save_path
        self.file_path=file_path        
        self.scale=scale
        self.n_cols=n_cols
        self.n_rows=math.ceil(len(os.listdir(self.file_path))/self.n_cols) 
        self.scale=space
        self.pad_val=pad_val
        self.space=space
        self.figsize=figsize
           
    def file_sorting(self):      
        import re,math,os
        '''
        function - 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表，按字母-数字顺序。因此数据文件内不要包含子文件夹，否则该文件夹名称也会被读取。文件名的格式为：'xx_100_xx.extension'
        '''
        dirs_files=os.listdir(self.file_path)
        dirs_files.sort()
        pattern=re.compile(r'[_](.*?)[_]', re.S) #re.compile(r'[_](.*?)[.]', re.S)
        fn_numExtraction=[(int(re.findall(pattern, fName)[0]),fName) for fName in dirs_files]
        #提取文件名中的数字，即聚类距离。并对应文件名
        fn_sort=sorted(fn_numExtraction) 
        fn_sorted=[i[1] for i in fn_sort]
        image_names=[] #存储的为图片的路径名
        for dir_file in fn_sorted:
            image_path=os.path.join(self.file_path, dir_file)
            if image_path.endswith('.png'):
                image_names.append(image_path)               
        q_imgs_paths=image_names[0:self.n_rows*self.n_cols] #保证提取的图片数量与所配置的n_rows*n_cols数量同    
        return q_imgs_paths
    
    def read_compress_imgs(self,imgs_fp):
        from PIL import Image
        import numpy as np
        '''
        function - 读取与压缩图片
        
        Paras:
        imgs_fp - 图像路径列表
        '''
        imgs=[] #存储的为读取的图片数据
        for img_fp in imgs_fp:
            img_array=Image.open(img_fp.rstrip())            
            img_resize=img_array.resize([int(self.scale * s) for s in img_array.size] ) #传入图像的数组，调整图片大小 
            img_trans=np.asarray(img_resize).transpose(2, 0, 1) #转置
            if (img_trans.shape[0] is not 3):
                img_trans=img_trans[0:3,:,:]
            imgs.append(img_trans)
            
        return imgs
    
    def make_dir(self):
        import os
        '''
        function - 建立文件夹，用于存储拼合的图片
        '''
        savefig_root=os.path.join(self.save_path,'imgs_combination')
        if os.path.exists(savefig_root):
            print("File exists!")
        else:
            os.makedirs(savefig_root)     
        return savefig_root
    
    def imgs_combination(self,imgs):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        '''
        function - 拼合图片
        '''
        #assert 断言是声明语句真假的布尔判断，如果值为真则执行程序，否则终止程序，避免运行过程中程序崩溃
        assert (imgs[0].ndim == 3) and (imgs[0].shape[0] == 3)
        assert len(imgs) <= self.n_rows * self.n_cols
        h, w=imgs[0].shape[1:]
        H=h * self.n_rows + self.space * (self.n_rows - 1)
        W=w * self.n_cols + self.space * (self.n_cols - 1)
        if isinstance(self.pad_val, np.ndarray): #isinstance（object，type）用于判断一个对象是否是一个已知类型
            self.pad_val=self.pad_val.flatten()[:, np.newaxis, np.newaxis]
        ret_img=(np.ones([3, H, W]) * self.pad_val).astype(imgs[0].dtype)
        for n, img in enumerate(imgs):
            r=n // self.n_cols
            c=n % self.n_cols
            h1=r * (h + self.space)
            h2=r * (h + self.space) + h
            w1=c * (w + self.space)
            w2=c * (w + self.space) + w
            ret_img[:, h1:h2, w1:w2] = img
        plt.figure(figsize=self.figsize)
        plt.imshow(ret_img.transpose(1,2,0))
        
        return ret_img
    
    def image_save(self,img,savefig_root):
        from PIL import Image
        import os
        '''
        function -保存图像
        '''
        if (img.shape[2] is not 3):
            img=img.transpose(1,2,0)
        Image.fromarray(img).save(os.path.join(savefig_root,'img_combination.jpg')) 
if __name__=="__main__":
    imgs_root='./graph/level'
    imgs_fp=glob.glob(os.path.join(imgs_root,'*.png'))
    imgs_save_path='./graph'
    n_cols=7;scale=0.3;space=1;pad_val=255;figsize=(30,10)
    combinePics=combine_pics(imgs_save_path,imgs_root,n_cols,scale,space,pad_val,figsize)
    imgs_compressed=combinePics.read_compress_imgs(imgs_fp)
    ret_img=combinePics.imgs_combination(imgs_compressed,)
    combinePics.image_save(ret_img,imgs_save_path)
