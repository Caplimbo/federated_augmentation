import argparse
DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba']
SIM_TIMES = ['small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-dataset',
    #                 help='name of dataset;',
    #                 type=str,
    #                 choices=DATASETS,
    #                 required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    default="CNN",
                    required=False)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=100)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=3)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=16)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)


    parser.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('--num-classes',
                        help='number of classes to classify',
                        type=int,
                        default=10)

    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=50)

    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=1e-3,
                    required=False)

    parser.add_argument('--tune-lr',
                help='learning rate for finetuning',
                type=float,
                default=1e-3,
                required=False)


    parser.add_argument('--threshold',
                    help='discriminator score lower bound on images generated',
                    type=float,
                    default=0.8,
                    required=False)

    parser.add_argument('-od', '--only-digits',
                        help='only use images of digits',
                        action='store_true')
    parser.add_argument('-t', '--use-tune',
                        help='tune before testing',
                        action='store_true')

    parser.add_argument('-a', '--augment',
                        help='whether we apply augmentation',
                        action='store_true')

    parser.add_argument('--report-interval',
                    help='report loss every ____ rounds;',
                    type=int,
                    default=50)


    parser.add_argument('-d', '--device', type=str, help='device to run the model', default='cuda', required=False)
    parser.add_argument('-s', '--save-dir', type=str, help='path to save the model and logs', default='checkpoints/run', required=False)

    return parser.parse_args()


