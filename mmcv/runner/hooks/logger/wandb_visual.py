# Copyright (c) Open-MMLab. All rights reserved.
import numbers

from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook

import wandb

@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 img_interval=1):
        """Initialize the WandB Logger Hook.

        Notes:
            img_interval is the interval of how often this logger hook is run.
            For example, if the interval is 10 and the img_interval is 1,
            then images are sent every 10 runner iterations. If the interval is
            10 and the img_interval is 10, then images are sent every 100 runner
            intervals.
        """
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag)
        self.calls = 0
        self.img_interval = img_interval

    @master_only
    def log(self, runner):
        """Takes metrics and images and logs them to weights and biases.

        If images are to be uploaded, then get_visualization must be a method
        within the head. It must return a list of dictionaries with each
        dictionary containing the following keys:

        'name': The name of the image(s)
        'image': Either a list of images, if it is to be uploaded as a JPG,
            or a single image if it is to be uploaded as a PNG file. The
            images themselves must be of the shape (h, w, c) where c is 1 if
            grayscale, 3 if RGB, or 4 if RGBA. Images can by numpy arrays or
            PIL Images.
        """
        metrics = {}
        for var, val in runner.log_buffer.output.items():
            if var in ['time', 'data_time']:
                continue
            tag = f'{var}/{runner.mode}'
            runner.log_buffer.output[var]
            if isinstance(val, numbers.Number):
                metrics[tag] = val
        metrics['learning_rate'] = runner.current_lr()[0]
        metrics['momentum'] = runner.current_momentum()[0]
        if metrics:
            wandb.log(metrics, step=runner.iter)


        get_vis = getattr(runner.model.module.bbox_head, 'get_visualization',
                          None)
        if callable(get_vis) and self.calls % self.img_interval == 0:
            images = get_vis(runner.model.module.last_vals['img'],
                             runner.model.module.CLASSES,
                             runner.model.module.test_cfg)
            for img in images:
                if isinstance(img['image'], list):
                    wandb.log({img['name']: [wandb.Image(x)
                                             for x in img['image']]},
                              step=runner.iter)
                else:
                    wandb.log({img['name']: wandb.Image(img['image'])},
                              step=runner.iter)

        self.calls += 1

    @master_only
    def after_run(self, runner):
        wandb.join()
