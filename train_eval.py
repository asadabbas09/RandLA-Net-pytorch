import time
import torch
import torch.nn.functional as F
import numpy as np
from blitz.utils.minibatch_weighting import minibatch_weight
import logging

def print_info(info, log_fp=None):
    message = ('Epoch: {}/{}, Duration: {:.2f}s '
               'Train Loss1: {:.3f}, Test Loss1:{:.3f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['train_loss1'], info['test_loss1'])
    '''message = ('Epoch: {}/{}, Duration: {:.3f}s, ACC: {:.4f}, '
               'Train Loss: {:.4f}, Test Loss:{:.4f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['acc'], info['train_loss'], info['test_loss'])'''
    print(message)
    if log_fp:
        with open(log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):

    y = y.cd.float()
    #for i in range(samples):
    #print([regressor(X)[1] for i in range(samples)])
    preds_d = [regressor(X)[1] for i in range(samples)]
    #print(np.shape(preds_d))
    preds = torch.stack(preds_d)
    #print(np.shape(preds))
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    #print(means, stds)

    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    #print(np.shape(ci_upper))
    #print(np.shape(ci_lower))
    #print(np.shape(y))

    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()

    # return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()
    return ic_acc, means, ci_lower, ci_upper, y

def run(model, train_loader, test_loader, num_nodes, epochs, optimizer,
        scheduler, device):
    iteration = 0
    for epoch in range(1, epochs + 1):


        t = time.time()
        train_loss1 = train(model, train_loader, optimizer, device, iteration)
        t_duration = time.time() - t
        scheduler.step()
        test_loss1 = test(model, test_loader, num_nodes, device)
        eval_info = {
            'train_loss1': train_loss1,
            # 'train_loss2': train_loss2,
            'test_loss1': test_loss1,
            # 'test_loss2': test_loss2,
            #'acc': acc,
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration,
            'lr': scheduler.get_lr()
        }

        print_info(eval_info)
        logging.info(eval_info)




def train(model, train_loader, optimizer, device, iteration):
    model.train()
    criterion = torch.nn.L1Loss()

    total_loss1 = 0
    total_loss2 = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data1= data.to(device)
        # x = data1.x.float()
        # print("X111", x.device)

        # data1 = data.to(f'cuda:{model.device_ids[0]}')
        # data1 = data.to("cuda:0,1")
        # print('inputs.device',data.get_device())
        # inputs = data.to('cuda:1')
        # inputs = data.to('cuda:2')
        # inputs = data.to('cuda:3')
        #x = data.x
        # print(next(model.parameters()).device)
        # print(data.y_node.float())#.device
        out_node = model(data1)#.cuda()

        loss1 = criterion(out_node[:,0], data1.y_node.float())
        # loss2 = F.mse_loss(out_graph[0], data.cd.float())
        # pi_weight = minibatch_weight(batch_idx=idx, num_batches=712)
        # loss1 = model.sample_elbo(inputs=data,
        #                    labels=data,#data.y_node.float(),
        #                    criterion=criterion,
        #                    sample_nbr=3,
        #                    complexity_cost_weight=pi_weight)

        '''loss2 = model.sample_elbo(inputs=data,
                           labels=data.cd.float(),
                           criterion=criterion,
                           sample_nbr=3)'''


        #print(loss1, loss2)
        # loss2 = loss2
        loss = loss1# + loss2
        #loss = loss2
        loss.backward()
        optimizer.step()
        total_loss1 += loss1.item()
        # total_loss2 += loss2.item()

        # iteration += 1
        # if iteration%10000==0:
        #     ic_acc, mean, under_ci_upper, over_ci_lower, y = evaluate_regression(model,
        #                                                                 data,
        #                                                                 data,
        #                                                                 samples=25,
        #                                                                 std_multiplier=1)

        #     #print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))

        #     #print("Loss: {:.4f}".format(loss))
        #     under_upper = under_ci_upper > y
        #     over_lower = over_ci_lower < y
        #     total = (under_upper == over_lower)

        #     print("Label {}".format(y), "Predicted {}".format(mean),
        #             "Upper Bound {}".format(under_ci_upper), "Lower Bound {}".format(over_ci_lower) )
    return total_loss1 / len(train_loader)#, total_loss2 / len(train_loader)


def test(model, test_loader, num_nodes, device):
    model.eval()
    criterion = torch.nn.L1Loss()
    correct = 0
    total_loss1 = 0
    total_loss2 = 0
    n_graphs = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data1 = data.to(device)
            # data1 = data.to(f'cuda:{model.device_ids[0]}')#.to(device)
            # data1 = data.to("cuda:0,1")#.to(device)
            # inputs = data.to('cuda:1')
            # inputs = data.to('cuda:2')
            # inputs = data.to('cuda:3')
            out_node = model(data1)


            loss1 = criterion(out_node[:,0], data1.y_node.float())
            # loss2 =  F.mse_loss(out_graph[0], data.cd.float()).item()
            # pi_weight = minibatch_weight(batch_idx=idx, num_batches=712)
            # loss1 = model.sample_elbo(inputs=data,
            #                    labels=data,
            #                    criterion=criterion,
            #                    sample_nbr=3,
            #                    complexity_cost_weight=pi_weight)

            '''loss2 = model.sample_elbo(inputs=data,
                               labels=data.cd.float(),
                               criterion=criterion,
                               sample_nbr=3)'''


            # loss2 = 10000*loss2

            total_loss1 += loss1.item()
            # total_loss2 += loss2
            #pred = out.max(1)[1]
            #correct += pred.eq(target).sum().item()
            #n_graphs += data.num_graphs
    return total_loss1 / len(test_loader)#, total_loss2 / len(test_loader)
