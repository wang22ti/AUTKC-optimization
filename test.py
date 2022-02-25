import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models

from utils import AverageMeter, evaluate, setup_seed
from dataset import data_loader

@torch.no_grad()
def validate(val_loader, model, k_list):
    topks =  [AverageMeter('Acc@%d' % k, ':6.2f') for k in k_list]
    autkcs = [AverageMeter('AUTKC@%d' % k, ':6.2f') for k in k_list]

    model = model.cuda()
    model.eval()
    for inputs, targets in val_loader:
        targets = targets.squeeze().cuda(non_blocking =True)
        inputs = inputs.float().cuda(non_blocking =True)
        with torch.no_grad():
            outputs = model(inputs).squeeze()
            precs, autkc = evaluate(outputs.data, targets, k_list)
            for _ in range(len(k_list)):
                topks[_].update(precs[_], inputs.size(0))
                autkcs[_].update(autkc[_], inputs.size(0))

    return [float(autkc.avg) for autkc in autkcs], [float(topk.avg) for topk in topks]

def test_and_save_model(model, test_loader, file_path, k_list):
    checkpoint = torch.load(file_path)
    if 'test_autkc' not in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
        autkc, prec = validate(test_loader, model, k_list)
        checkpoint['test_autkc'], checkpoint['test_precs'] = autkc, prec
        torch.save(checkpoint, file_path)
        return autkc, prec
    else:
        return checkpoint['test_autkc'], checkpoint['test_precs']

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset = 'place-365'
dataset_dir = '{}'.format(dataset)
checkpoint_dir = ''

setup_seed(int(checkpoint_dir.split('_')[-1]))
k_list = [_ for _ in range(1, 11)]
_, _, test_loader, num_class = data_loader(dataset_dir, 256, 4)

if dataset in ['place-365', ]:
    model = nn.Sequential(
        nn.Linear(2048, 512, bias=True), 
        nn.ReLU(),
        nn.Linear(512, 256, bias=True), 
        nn.ReLU(),
        nn.Linear(256, num_class, bias=True), 
    )
else:
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)

test_results = list()
for k in tqdm(k_list):
    file_name = 'best_autkc{}.pth'.format(k)
    file_path = os.path.join(checkpoint_dir, file_name)
    autkc, prec = test_and_save_model(model, test_loader, file_path, k_list)
    test_results.append([file_name, ] + autkc + prec)

    file_name = 'best_prec{}.pth'.format(k)
    file_path = os.path.join(checkpoint_dir, file_name)
    autkc, prec = test_and_save_model(model, test_loader, file_path, k_list)
    test_results.append([file_name, ] + autkc + prec)

for result in test_results:
    print(result[0].ljust(16), ' '.join(['%.2f' % _ for _ in result[1:]]), sep='\t')
