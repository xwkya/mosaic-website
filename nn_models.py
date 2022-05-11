import torch
import torch.nn
import torch.nn.functional as F
from torchvision import transforms as T
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4, P4MConvP4M, P4MConvZ2
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

class NNModel(torch.nn.Module):
    def __init__(self, out_features, name_to_index, name):
        super(NNModel, self).__init__()
        self.name = name
    
    def save(self, number=None):
        if number != None:
            torch.save(self.state_dict(), "models/"+self.name+str(number)+".pt")
        else:
            torch.save(self.state_dict(), "models/"+self.name+".pt")
    
    def load(self):
        try:
            self.load_state_dict(torch.load("models/"+self.name+".pt", map_location=torch.device('cpu')))
        except Exception as e:
            print(e)
            self.load_state_dict(torch.load("models/"+self.name+"1500.pt"))
    
    def forward(self):
        raise NotImplementedError


class NNpolicy_torchresize(NNModel):
    def __init__(self, out_features, name_to_index, name):
        self.name_to_index = name_to_index
        self.out_features = out_features
        super(NNpolicy_torchresize, self).__init__(out_features, name_to_index, name)

        # Input of size (bsz, 3, 64, 64)
        # (bsz, 3, 32, 32)
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        # (32, 32, 32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (32, 16, 16)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        # (64, 16, 16)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (64, 8, 8)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        # (64, 8, 8)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # (128, 4, 4) -> 2048
        self.BN1 = torch.nn.BatchNorm1d(1024)
        self.BN2 = torch.nn.BatchNorm1d(1024)

        self.linear1 = torch.nn.Linear(2048, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, out_features)

    def forward(self, x):
        # (bsz, 256, 256, 3)
        x = torch.transpose(x, 1, 3)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), 2048)
        x = F.relu(self.linear1(x))
        x = self.BN1(x)
        h = x
        x = F.relu(self.linear2(x)) + h
        x = self.BN2(x)
        h = x
        x = F.relu(self.linear3(x))
        
        #x = F.softmax(x, dim = -1)
        return x

class NNGCNN(NNModel):
    def __init__(self, out_features, out_groups, name_to_index, name, predict_group=True):
        super(NNGCNN, self).__init__(out_features, name_to_index, name)
        self.out_features = out_features
        self.out_groups = out_groups
        self.name_to_index = name_to_index

        self.conv1 = P4MConvZ2(3, 32, kernel_size=3, padding=1)
        self.conv2 = P4MConvP4M(32, 64, kernel_size=3, padding=1)
        self.conv3 = P4MConvP4M(64, 256, kernel_size=3, padding=1)
        self.predict_group = predict_group
        if self.predict_group:
            self.fc1 = torch.nn.Linear(4*4*256*8, 1024)
        else:
            self.fc1 = torch.nn.Linear(4*4*256, 1024)
        self.fc2 = torch.nn.Linear(1024, 8192)
        
        if predict_group:
            self.fc_out = torch.nn.Linear(8192, out_features*out_groups)
        else:
            self.fc_out = torch.nn.Linear(8192, out_features)

        

    def forward(self, x):
        x = torch.transpose(x, 1, 3)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = plane_group_spatial_max_pooling(x, 2, 2)

        if not self.predict_group:
            x = torch.mean(x, dim=2)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x_features = self.fc_out(x)

        return x_features
        

class NNGCNN2(NNModel):
    def __init__(self, out_features, out_groups, name_to_index, name):
        super(NNGCNN2, self).__init__(out_features, name_to_index, name)
        self.out_features = out_features
        self.out_groups = out_groups
        self.name_to_index = name_to_index

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(4*4*128, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 2048)
        self.fc_groups = torch.nn.Linear(2048, out_groups)
        self.fc_features = torch.nn.Linear(2048, out_features)

    def forward(self, x):
        x = torch.transpose(x, 1, 3)

        x = F.relu(self.conv1(x))
        x = torch.nn.AvgPool2d(2, 2)(x)
        x = F.relu(self.conv2(x))
        x = torch.nn.AvgPool2d(2, 2)(x)
        x = F.relu(self.conv3(x))
        x = torch.nn.AvgPool2d(2, 2)(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x_features = self.fc_features(x)
        x_groups = self.fc_groups(x)
        return x_features, x_groups

class NNConstructed(NNModel):
    def __init__(self, out_features, name_to_index, name, batchNorm2d=False, batchNorm1d=False, linear_size=[1024, 1024, 2048], conv_channels=[32, 64, 128]):
        self.name_to_index = name_to_index
        self.out_features = out_features
        super(NNConstructed, self).__init__(out_features, name_to_index, name)

        # Input of size (bsz, 3, 64, 64)
        # (bsz, 3, 32, 32)
        self.batchNorm1d = batchNorm1d
        self.batchNorm2d = batchNorm2d
        conv_list = []
        linear_list = []
        norms2d_list = []
        norms1d_list = []
        pool_list = []

        conv_list.append(torch.nn.Conv2d(3, conv_channels[0], 3, padding=1))
        pool_list.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        norms2d_list.append(torch.nn.BatchNorm2d(conv_channels[0]))
        for i in range(len(conv_channels)-1):
            conv_list.append(torch.nn.Conv2d(conv_channels[i], conv_channels[i+1], 3, padding=1))
            pool_list.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            norms2d_list.append(torch.nn.BatchNorm2d(conv_channels[i+1]))
        
        size = int(32/(2**len(pool_list)))
        linear_list.append(torch.nn.Linear(conv_channels[-1]*size*size, linear_size[0]))
        norms1d_list.append(torch.nn.BatchNorm1d(linear_size[0]))
        for i in range(len(linear_size)-1):
            linear_list.append(torch.nn.Linear(linear_size[i], linear_size[i+1]))
            norms1d_list.append(torch.nn.BatchNorm1d(linear_size[i+1]))
        linear_list.append(torch.nn.Linear(linear_size[-1], out_features))

        self.linear_list = torch.nn.ModuleList(linear_list)
        self.conv_list = torch.nn.ModuleList(conv_list)
        self.norms2d_list = torch.nn.ModuleList(norms2d_list)
        self.norms1d_list = torch.nn.ModuleList(norms1d_list)
        self.pool_list = torch.nn.ModuleList(pool_list)

    def forward(self, x):
        
        x = torch.transpose(x, 1, 3)

        for conv_layer, norm_layer, pool_layer in zip(self.conv_list, self.norms2d_list, self.pool_list):
            x = F.relu(conv_layer(x))
            if self.batchNorm2d:
                x = norm_layer(x)
            
            x = pool_layer(x)
        
        x = x.view(x.size(0), -1)

        l = 0
        for linear_layer, norm_layer in zip(self.linear_list, self.norms1d_list):
            
            l = l+1
            x = linear_layer(x)
            x = F.relu(x)
            
            if self.batchNorm1d:
                x = norm_layer(x)
            
        x = self.linear_list[-1](x)
            
        return x    
