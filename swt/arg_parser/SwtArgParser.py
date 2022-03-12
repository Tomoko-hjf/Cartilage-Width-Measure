'''
Author: hjf
Date: 2022-02-20 12:03:46
LastEditTime: 2022-02-20 15:16:08
Description:  命令行参数读取器
'''
import argparse


class SwtArgParser(argparse.ArgumentParser):
    """Argument parser for running orientation detection model."""

    def __init__(self):
        super().__init__()

        self.add_argument('--image', default='E:/hjf/work/tools/stroke-width-transform/images/text.jpg', metavar='IMAGE',
                          help='The image file to process.')

        self.add_argument('--bright_on_dark', default=False, action='store_true',
                          help='Enables bright on dark selection.')

        #self.add_argument(
        #    "--validation_dir", "-vd", default="/tmp",
        #    help="[default: %(default)s] The location of the validation data.",
        #    metavar="<VD>",
        #)

        #self.set_defaults(
        #    validation_dir=os.path.join('dataset', 'eval')
        #    )
