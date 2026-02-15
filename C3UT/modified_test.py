"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch

def set_thread_num(num_threads: int = 16):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)  # 限制 OpenMP 线程数
    os.environ["MKL_NUM_THREADS"] = str(num_threads)  # 限制 MKL 线程数
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)  # 限制 numexpr 线程数
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)  # 限制 vecLib 线程数
    os.environ["BLIS_NUM_THREADS"] = str(num_threads)  # 限制 BLIS 线程数
    torch.set_num_threads(num_threads)
    
if __name__ == '__main__':
    set_thread_num(32)
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    save_buffer = []
    path_buffer = []
    buffer_size = 50  # 每20张图片一起保存
    
    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
    
        save_buffer.append(visuals)
        path_buffer.append(img_path)
    
        if i % buffer_size == 0 and i > 0:
            print(f'Batch saving at {i}...')
            for visuals, img_path in zip(save_buffer, path_buffer):
                save_images(webpage, visuals, img_path, width=opt.display_winsize)
                del visuals
            save_buffer.clear()
            path_buffer.clear()
            torch.cuda.empty_cache()
    
    # 保存剩余未写的
    for visuals, img_path in zip(save_buffer, path_buffer):
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()
