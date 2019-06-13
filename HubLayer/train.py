import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import tables as tb
import pickle

PATH = os.getcwd()
# Physical parameters
m = 1.
L = 1.
g = 1. 

# largest p and q can be.    
pmax = 4.
thetamax = np.pi/2.1

## variant of Pendulum Net
class PendulumNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(PendulumNet, self).__init__()
        self.input = nn.Linear(D_in, H)
        self.fc1 = nn.Linear(H, H)
        self.fc2 = nn.Linear(H, H)
        self.out = nn.Linear(H, D_out)

    def forward(self, X):
        X = torch.sin(self.input(X))
        X = torch.sin(self.fc1(X))
        X = torch.sin(self.fc2(X))
        X = self.out(X)
        return X 

def random_uniform_phase_space(thetamax, pmax, numOfParticles):
    thetais = np.random.uniform(-thetamax, thetamax, numOfParticles)
    ps = np.random.uniform(-pmax,pmax, numOfParticles)
    return thetais, ps

def train(args, model, device, optimizer, scheduler, epochs_trained, existing_model):
    regularize_energy = False
    
    batch_size = args.batch_size
    n_train = args.num_of_particles * args.num_of_samples_per_particles
    num_epochs = args.num_of_epochs
    chunks = args.chunks

    criterion = nn.MSELoss()
    lossdata = []
 
    t0 = time.time()
    ''' 
    h5file = tb.open_file(PATH + '/experiments/' + args.model_name + '/loss.h5', mode='a', title="loss history")
    root = h5file.root
    x = h5file.createCArray(root,'x',tb.Float64Atom(),shape=(1, num_epochs))
    '''

    for epoch in range(args.num_of_epochs):       
        '''
        np.random.seed(0)
        tt = np.random.uniform(0, args.time, n_train)
        tt = torch.Tensor(tt.reshape(-1,1)).to(device=device)
        tt.requires_grad_(True)

        theta0, p0 = random_uniform_phase_space(thetamax, pmax, args.num_of_particles)
        theta0, p0 = np.repeat(theta0, args.num_of_samples_per_particles), np.repeat(p0, args.num_of_samples_per_particles)
        theta0, p0 = torch.Tensor(theta0.reshape(-1,1)).to(device=device), torch.Tensor(p0.reshape(-1,1)).to(device=device)

        X = torch.cat((theta0,p0,tt),dim=1).to(device=device)
        '''
        np.random.seed(0)
        shuffle_index = torch.randperm(n_train)

        tt_ = np.random.uniform(0, args.time, n_train)
        tt_ = torch.Tensor(tt_.reshape(-1,1))
    
        tt = tt_[shuffle_index].to(device=device)
        tt.requires_grad_(True)
    
        theta0_, p0_ = random_uniform_phase_space(thetamax, pmax, args.num_of_particles)
        theta0_, p0_ = np.repeat(theta0_, args.num_of_samples_per_particles), np.repeat(p0_, args.num_of_samples_per_particles)
        theta0_, p0_ = torch.Tensor(theta0_.reshape(-1,1)), torch.Tensor(p0_.reshape(-1,1))
        theta0, p0 = theta0_[shuffle_index].to(device=device), p0_[shuffle_index].to(device=device)
        
    
        X_ = torch.cat((theta0,p0,tt),dim=1)
    
        X = X_.to(device=device)

        for i in range(int(len(tt)/args.batch_size)):

            Xi = X[i*batch_size:(i+1)*batch_size]

            theta0i = theta0[i*batch_size:(i+1)*batch_size]
            p0i = p0[i*batch_size:(i+1)*batch_size]

            tti = tt[i*batch_size:(i+1)*batch_size]

            # forward
            theta = theta0i + model(Xi)[:,0].clone().reshape(-1,1) * tti
            p = p0i + model(Xi)[:,1].clone().reshape(-1,1) * tti


            p.retain_grad()
            theta.retain_grad()

            H = p**2/(2*m*L**2) + m*g*L*(1 - torch.cos(theta))

            ## use auto diff for d_H_d_p and d_H_d_theta
            d_H_d_p, = torch.autograd.grad(H, p, H.new(H.shape).fill_(1),create_graph = True)
            d_H_d_theta, = torch.autograd.grad(H, theta, H.new(H.shape).fill_(1),create_graph = True)

            d_theta_d_t, = torch.autograd.grad(theta, tti, theta.new(theta.shape).fill_(1),create_graph = True)
            d_p_d_t, = torch.autograd.grad(p, tti, p.new(p.shape).fill_(1),create_graph = True)

            loss_eqn = criterion(d_p_d_t, - d_H_d_theta) + criterion(d_theta_d_t, d_H_d_p)
            
            
            loss = loss_eqn
            # optimize
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

        # save lossdata each epoch
        lossdata.append(float(loss.data))

        if(epoch ==0):
            print('epoch [{}/{}], loss eqns of motion:{:.16f}'.format(epoch + 1, num_epochs, loss_eqn.data))
            print('Trainng time for one epoch: ' + str(time.time() - t0))
            print('Estimated total training time: '+str((time.time() - t0) * num_epochs/(60.0)))
            if(regularize_energy == True):
                print('epoch [{}/{}], loss energy:{:.16f}'.format(epoch + 1, num_epochs, loss_energy.data))
                        
        if( (epoch+1) % (num_epochs/chunks)==0):
            print('epoch [{}/{}], loss eqns of motion:{:.16f}'.format(epoch + 1, num_epochs, loss_eqn.data))
            if(regularize_energy == True):
                print('epoch [{}/{}], loss energy:{:.16f}'.format(epoch + 1, num_epochs, loss_energy.data))

            # check point
            torch.save({
                    'epochs_trained': epoch + epochs_trained,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, PATH + '/experiments/'+args.model_name+'/model_state')
            
            # save loss history
            np.save(PATH + '/experiments/' + args.model_name+'/loss.npy', np.array(lossdata))
            print('Check Point at epoch:'+str(epoch+1))       
        # annealing learning rate
        scheduler.step()



def main():
    # Device
    print('Running with GPU: '+str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type(torch.DoubleTensor)


    parser = argparse.ArgumentParser(description='Integration for Pendulum Motion')
    parser.add_argument('--neurons-per-layer', type = int, default=32, metavar='numOfNeuronsPerLayer',help = 'neurons per layer; default = 32')
    #parser.add_argument('--num-of-hidden-layers', type = int, default=2, metavar='numOfHiddenLayers',help = 'number of hidden layers; default = 2')
    parser.add_argument('--num-of-particles', type = int, default=1, metavar = 'numOfParticles',help='num of particles; default = 1')
    parser.add_argument('--num-of-samples-per-particles', type = int, default=8192, metavar = 'numOfSamplesPerParticle',help='num of particles; default = 8192')

    parser.add_argument('--num-of-epochs', type = int, default=5000, metavar = 'numOfEpochs',help='num of epochs to train; default = 5000')
    parser.add_argument('--batch-size', type = int, default=512, metavar = 'batchSize',help='batch size ; default = 512')
    parser.add_argument('--chunks', type = int, default=100, metavar = 'chunks',help='divisible by num_epochs, # chunks to save data; default = 100')

    parser.add_argument('--learning-rate', type = float, default=1e-4, metavar = 'lr', help = 'starting learning rate; default = 1e-4')
    parser.add_argument('--model-name')
    parser.add_argument('--time', type = float, default = 1.0, metavar='T', help = 'integrationTime')

    parser.add_argument('--lr-decay-factor', type = float, default = 0.99, metavar = 'gamma', help = 'decay factor rate; default = 0.99')
    parser.add_argument('--lr-update-step', type = int, default = 100, metavar = 'gamma', help = 'lr update after n epochs; default = 100')

    args = parser.parse_args()


    # initialize model
    model_pendulum = PendulumNet(3, args.neurons_per_layer, 2)
    epochs_trained = 0

    # check if model already exist
    existing_model = os.path.exists(PATH+'/experiments/'+args.model_name)
    if(existing_model):
        confirm = input("Confirm to train the existing model "+args.model_name+" :\n")
        if(confirm=="y" or confirm == "yes"):
            checkpoint = torch.load(PATH+'/experiments/'+args.model_name+'/model_state')
            model_pendulum.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs_trained = checkpoint['epochs_trained']
            
        else:
            return
    else:
        os.mkdir(PATH + '/experiments/'+args.model_name)
        print('Train a new model;')
        pickle.dump( args, open( PATH + '/experiments/'+args.model_name+"/args.p", "wb" ) )
    
    optimizer = torch.optim.Adam(model_pendulum.parameters(), lr = args.learning_rate, weight_decay = 2e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.lr_update_step, gamma=args.lr_decay_factor)
    
    model_pendulum = model_pendulum.to(device = device)
    train(args, model_pendulum, device, optimizer, scheduler, epochs_trained, existing_model)
    
    print("Training was sucessful.")
 
if __name__ == '__main__':
    main()

