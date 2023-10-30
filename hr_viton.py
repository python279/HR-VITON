import torch.nn as nn

from cp_dataset_test import CPDatasetTest, CPDataLoader

from networks import ConditionGenerator
from network_generator import SPADEGenerator
from utils import *

from PIL import Image
from test_generator import test, get_opt, load_checkpoint_G, load_checkpoint
import sys
import tempfile
import json
import traceback
import shutil
from copy import deepcopy


class HrViton(object):
    def __init__(self):
        tmpdir = tempfile.mkdtemp()
        sys.argv = ['test_generator.py', '--occlusion', '--test_name', 'test', '--dataroot', tmpdir, '--data_list', 'test.txt', '--datamode', 'test', '-j', '1', '--output_dir', tmpdir]

        opt = get_opt()
        self.opt = deepcopy(opt)

        # tocg
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
        tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)

        # generator
        opt.semantic_nc = 7
        generator = SPADEGenerator(opt, 3+3+3)

        # Load Checkpoint
        load_checkpoint(tocg, opt.tocg_checkpoint, opt)
        load_checkpoint_G(generator, opt.gen_checkpoint, opt)

        self.tocg = tocg
        self.generator = generator

    def __call__(self, inputs: dict):
        try:
            input_tmpdir = tempfile.mkdtemp()
            output_tmpdir = tempfile.mkdtemp()
            self.opt.dataroot = input_tmpdir
            self.opt.output_dir = output_tmpdir

            image_id, cloth_id = set([]), set([])
            for input in inputs:
                if input['name'] == 'cloth':
                    cloth_id.add(input['id'])
                if input['name'] == 'image':
                    image_id.add(input['id'])
                if input['type'] == 'image':
                    dir_path = os.path.join(input_tmpdir, "test", input['name'])
                    os.makedirs(dir_path, exist_ok=True)
                    input['data'].save(os.path.join(dir_path, input['filename']))
                elif input['type'] == 'json':
                    dir_path = os.path.join(input_tmpdir, "test", input['name'])
                    os.makedirs(dir_path, exist_ok=True)
                    with open(os.path.join(dir_path, input['filename']), 'w') as f:
                        json.dump(input['data'], f)

            # 从 image_id 和 cloth_id 中取各自特有的元素
            _cloth_id = list(cloth_id - image_id)[0]
            _image_id = list(image_id)[0]
            with open(os.path.join(input_tmpdir, 'test.txt'), 'w') as f:
                f.write('{}.jpg {}.jpg\n'.format(_image_id, _cloth_id))

            test_dataset = CPDatasetTest(self.opt)
            test_loader = CPDataLoader(self.opt, test_dataset)
            test(self.opt, test_loader, self.tocg, self.generator)

            output_image = Image.open(os.path.join(output_tmpdir, '{}_{}.png'.format(_image_id, _cloth_id)))
        except Exception:
            output_image = None
            print(traceback.format_exc())
        finally:
            shutil.rmtree(input_tmpdir)
            shutil.rmtree(output_tmpdir)
            pass

        return output_image
