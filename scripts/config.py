import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--teacher_models', required=True, nargs="+", help='name of the ensembled model to distill')
parser.add_argument('--output', type=str, help='directory of output')
parser.add_argument('--save_epoch', type=int, help='how many epochs to save the model')

# distilled model architecture
parser.add_argument('--student_arch', type=str, choices=["Default2018", "Dense"], help='student model architecture')

# training data
parser.add_argument('--trligte', required=True, help='location of training ligand cache file input')
parser.add_argument('--trrecte', required=True, help='location of training receptor cache file input')
parser.add_argument('--trainfile', required=True, help='location of training information')

# validation data
parser.add_argument('--reduced_test', help='reduced test set')

# testing data
parser.add_argument('--teligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--terecte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information')

# training settings
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--dropout', '-d',default=0, type=float,help='dropout of layers')
parser.add_argument('--weight_decay',default=0.001,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--T',default=1,type=float,help='temperature for softmax function')
parser.add_argument('--clip',default=10,type=float,help='keep gradients within [clip]')
parser.add_argument('--epochs',default=10000,type=int,help='epoch number for training the student model')
parser.add_argument('--batch_size',default=50,type=int,help='batch size for training')

# lr decreasing settings
parser.add_argument('--step_when',type=int,help='iterations with no auc increasment')
parser.add_argument('--step_end_cnt',type=int,help='times of decreasing learning rate')
parser.add_argument('--use_weight', default=None, help="load a model")

# wandb settings
parser.add_argument('--job_name', help='name of the job shown in wandb')
parser.add_argument('--project', help='wandb project')
parser.add_argument('--user', default=None, help='wandb user name')

args = parser.parse_args()