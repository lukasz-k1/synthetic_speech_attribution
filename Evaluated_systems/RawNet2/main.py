import argparse
import sys
import os
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils import gen_train_eval,Dataset_SPCUP2022_train,Dataset_SPCUP2022_eval
from model import RawNet
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    
    save_path_predict=os.path.join(save_path,'predictions.csv')
    save_path_scores=os.path.join(save_path,'scores.csv')
    with open(save_path_predict, 'w') as fh:
        fh.write('Fielename, Predicted label\n')
        fh.close() 

    with open(save_path_scores, 'w') as fh:
        fh.write('Fielename, class 0, class 1, class 2, class 3, class 4, class 5\n')
        fh.close() 
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out
                      ).data.cpu().numpy()#.ravel()

        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path_predict, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                predict=0
                cm = np.exp(cm)
                predict = cm.tolist().index(max(cm))

                fh.write('{},{}\n'.format(f, predict))
        fh.close()   
    


        with open(save_path_scores, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                cm = np.exp(cm)
                fh.write('{},{},{},{},{},{},{}\n'.format(f, cm[0], cm[1], cm[2], cm[3], cm[4], cm[5]))
        fh.close() 
    
    print('Predictions saved to {}'.format(save_path_predict))
    print('Scores saved to {}'.format(save_path_scores))


def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    #set objective (Loss) functions
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss()#weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t epoch train_accuracy: {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SPCUP 2022 system')
    # Dataset
    parser.add_argument('--DATA_PATH', type=str, default='spcup_2022_training_part1/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same DATA_PATH directory.')

    parser.add_argument('--CSV_LABELS_PATH', type=str, default='spcup_2022_training_part1/labels.csv', help='Change with path to user\'s csv labels directory address')

    parser.add_argument('--random_state', type=int, default=42, help='Random state of train test split')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', default=False,
                        help='eval mode')
    parser.add_argument('--eval_1', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    

    dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    # dir_yaml = 'model_config_RawNet.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    train_filenames, eval_filenames, train_labels, eval_labels = gen_train_eval(args.CSV_LABELS_PATH, random_state=args.random_state)

    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
        args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    # device = 'cuda'
    
    #model 
    model = RawNet(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    #evaluation 
    if args.eval:
        eval_set=Dataset_SPCUP2022_eval(list_IDs = eval_filenames,base_dir = args.DATA_PATH)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    #evaluation 1
    if args.eval_1:
        eval_filenames=[]
        for (dirpath, dirnames, filenames) in os.walk(args.DATA_PATH):
          eval_filenames.extend(filenames)
        
        if 'labels_eval_part2.csv' in eval_filenames:
            eval_filenames.remove('labels_eval_part2.csv')

        if 'labels_eval_part1.csv' in eval_filenames:
            eval_filenames.remove('labels_eval_part1.csv')
        
        if 'labels.csv' in eval_filenames:
            eval_filenames.remove('labels.csv')

        eval_set=Dataset_SPCUP2022_eval(list_IDs = eval_filenames,base_dir = args.DATA_PATH)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

     
    # define train dataloader
    train_set=Dataset_SPCUP2022_train(list_IDs = train_filenames,labels = train_labels,base_dir = args.DATA_PATH)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True)
    
    del train_set
    
    # define validation dataloader
    dev_set = Dataset_SPCUP2022_train(list_IDs = eval_filenames,
		labels = eval_labels,
		base_dir = args.DATA_PATH)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 96
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - loss: {} - train_accuracy: {:.2f} - valid_accuracy: {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        
        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
        best_acc = max(valid_accuracy, best_acc)
        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
