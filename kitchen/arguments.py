import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sub_skill_cat', type=int, default=4)
    parser.add_argument('--suite', type=str, default='microwave_train', help='[microwave_train, kettle_train, slider_train, topburner_train]')
    parser.add_argument('--batch_size', type=int, default=20000) 
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--tb_dir', type=str, default='tensorboard')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_index', type=int, default=7)
    parser.add_argument('--continuous_coefficient', type=float, default=1)
    parser.add_argument('--fast_interations', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=40000)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--update_lr', type=float, default=0.0001)

    parser.add_argument('--test_update_lr', type=float, default=0.001)
    parser.add_argument('--test_skill_num', type=int, default=4)
    parser.add_argument('--test_suite', type=str, default='microwave-kettle-topburner-light', help=['microwave-kettle-topburner-light', 'microwave-bottomburner-light-slider', 'microwave-kettle-hinge-slider', 'microwave-kettle-hinge-slider'])
    return parser.parse_args()
