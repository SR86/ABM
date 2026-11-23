import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
    # parser.add_argument('--ratio', type=float, default=0.1, help='ratio of training data')
    # parser.add_argument('--PoisionRatio', type=float, default=0.01, help='ratio of training data')
    parser.add_argument('--model_num', type=int, default=1, help='The number of models to train')

    #Storage path
    parser.add_argument('--datasetPath', type=str, default="./data/", help='dataset storage path')
    parser.add_argument('--outModelPath', type=str, default="./outModel/", help='outModel storage path')
    parser.add_argument('--outDataPath', type=str, default="./outData/", help='data storage path')

    # training hyper parameters
    parser.add_argument('--batch_size', type=int, default=120, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--cuda', type=int, default=1)

    # backdoor attacks by poision attack
    # parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='BlendTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')

    #Bab
    parser.add_argument('--inject_portion_defend', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label_defend', type=int, default=10, help='class of target label')
    parser.add_argument('--trigger_type_defend', type=str, default='AnotherBadNetTrigger',
                        help='type of backdoor trigger')


    return parser

    # ['SIGTrigger', 'BadNetTrigger', 'BlendTrigger', 'CBATrigger','RefoolTrigger','AnotherBadNetTrigger','BppTrigger','NarTrigger','WaNetTrigger']
