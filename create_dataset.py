import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from correspondence_bayes_car import run, Net, pointNet
from utils_bayes_car import preprocess_spiral

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import meshio
import json
import logging
os.makedirs('log', exist_ok=True)
# from datasets import FAUST

parser = argparse.ArgumentParser(description='shape correspondence')
parser.add_argument('--dataset', type=str, default='/media/asad/Work/Documents/Emulator/AustalDataset/extracted/')
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--seq_length', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--samples', type=int, default=100)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--create_dataset',  action='store_true')
parser.add_argument('--resume',  action='store_true')
parser.add_argument('--pretrained',  action='store_true')
parser.add_argument('--pointnet',  action='store_true')
parser.add_argument('--exp',  type=str, default='exp1')
args = parser.parse_args()
torch.set_num_threads(args.n_threads)
logging.basicConfig(level=logging.INFO, filename=f'log/ships_samples_{args.samples}_length_{args.seq_length}_dilation_{args.dilation}_exp{args.exp}.log', filemode='w',format='%(message)s')

from conv import SpiralConv
class customModel(nn.Module):
	def __init__(self, pretrained):
		super(customModel, self).__init__()
		self.fc0 = pretrained[0]
		self.conv1 = pretrained[1]
		self.conv2 = pretrained[2]
		self.conv3 = pretrained[3]
		# self.conv4 = pretrained[4]
		self.sigmoid = torch.nn.Sigmoid()
#         self.conv5 = pretrained[5]
#         self.conv5 = SpiralConv(256, 256, 10)
		# self.fc1 = pretrained[5]#nn.Linear(256, 512)
		# self.fc2 = pretrained[6]#nn.Linear(512, 1)
		self.fc1 = nn.Linear(128, 256)
		self.fc2 = nn.Linear(256, 1)

	def forward(self, data):
		x = data.x.float()
		indices = data.indices.long()
		x = self.sigmoid(self.fc0(x))
		x = self.sigmoid(self.conv1(x, indices))
		x = self.sigmoid(self.conv2(x, indices))
		x = self.sigmoid(self.conv3(x, indices))
		# x = self.sigmoid(self.conv4(x, indices))
		x = self.sigmoid(self.fc1(x))
		x = F.dropout(x,p=0.5, training=self.training)
		out_node = (self.fc2(x))

		return out_node#, out_graph



class MyDataset(InMemoryDataset):
	def __init__(self, root,  transform=None, pre_transform=None):
		super(MyDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])
		# mesh_list = mesh_list
	@property
	def raw_file_names(self):
		return []
	@property
	def processed_file_names(self):
		return ['bike.pt']

	def download(self):
		pass

	def process(self):

		data_list = []
		#for mesh in mesh_list:
		for i in range (len(mesh_list1)):
			print(i)
			try:

				mesh_io = meshio.read(mesh_list1[i])
				# print(mesh_list[i])

				pos = torch.from_numpy(mesh_io.points.astype(np.float32)).to(torch.float)
				face = torch.from_numpy(mesh_io.cells[0].data.astype(np.float32)).to(torch.long).t().contiguous()

				# print(face.shape, pos.shape)

				f = mesh_io.cells[0].data#.astype(np.float32)
				#print(f.shape[1])
				spiral = preprocess_spiral(f, args.seq_length, dilation=args.dilation)
				cx = drag_list_cx1[i]
				cy = drag_list_cy1[i]
				#print(spiral)


				indices = spiral.to(torch.long).contiguous()


				#edge_index = torch.cat([tetra[:2], tetra[1:3, :], tetra[-2:], tetra[::2], tetra[::3], tetra[1::2]], dim=1)
				#print(np.shape(mesh.point_data['point_scalars']))
				#y_node = torch.from_numpy(mesh.point_data['point_scalars'][:,0])
				y_node = torch.from_numpy(mesh_io.point_data['p'].astype(np.float32))
				y_stress = torch.from_numpy(mesh_io.point_data['wallShearStress'].astype(np.float32))




				'''f = open(drag_list[i], 'r')
				cd = f.read()
				cd = np.asarray(float(cd))
				cd = torch.from_numpy(cd).to(torch.float)
				f.close'''


				data = Data(x=pos, face=face, y_node = y_node, y_stress = y_stress, indices = indices, cx = cx, cy = cy)

				data_list.append(data)
			except Exception as e:
				print(e,'Error in ', mesh_list1[i])


		if self.pre_filter is not None:
			data_list = [data for data in data_list if self.pre_filter(data)]

		if self.pre_transform is not None:
			data_list = [self.pre_transform(data) for data in data_list]

		print(np.shape(data_list))
		data, slices = self.collate(data_list)
		#print(np.shape(data), np.shape(slices))
		torch.save((data, slices), self.processed_paths[0])


def meshfile_list(path_dir):
		keyword_p = 'vessel.vtk'
		keyword_d = 'drag_coeffs.json'
		mesh_list = []
		drag_list_cx = []
		drag_list_cy = []
		# print(path_dir)
		for root, dirs, files in os.walk(path_dir):
			print(root, dirs, files)
			path = root.split(os.sep)
			for file in files:
				if keyword_p in file:
					mesh_list.append(root+'/'+file)
					print(root+'/'+file, " has the keyword p")
				if keyword_d in file:
					with open(root+'/'+file) as json_file:
						data = json.load(json_file)
						drag_list_cx.append(data['cx'])
						drag_list_cy.append(data['cy'])
					print(root+'/'+file, " has the keyword p")

		return mesh_list, drag_list_cx, drag_list_cy
# print(f"Looking for files in {args.dataset}")




if (args.create_dataset):



	print(f"Looking for files in {args.dataset}")

	global mesh_list1
	global drag_list_cx1
	global drag_list_cy1
	list1, cx, cy = meshfile_list(args.dataset)#
	# print(list1)
	mesh_list1 = list1[:args.samples]
	drag_list_cx1 = cx[:args.samples]
	drag_list_cy1 = cy[:args.samples]

	print(f"creating dataset for {args.samples}")

	pre_transform = T.Compose([T.FaceToEdge(remove_faces=False), T.Constant(value=1)])

	# dataset = MyDataset(f'processdata/ships_{args.samples}_{args.seq_length}_{args.dilation}', pre_transform=T.FaceToEdge(remove_faces=False))
	dataset = MyDataset(f'processdata/ships_{args.samples}_{args.seq_length}_{args.dilation}')

	torch.save(dataset, f'torchdata/ships_samples_{args.samples}_length_{args.seq_length}_dilation_{args.dilation}_cxy.pt')

else:



	print("Loading Dataset")

	dataset = torch.load(f'torchdata/ships_samples_{args.samples}_length_{args.seq_length}_dilation_{args.dilation}.pt')
	logging.info(f'torchdata/ships_samples_{args.samples}_length_{args.seq_length}_dilation_{args.dilation}_exp{args.exp}.pt')
	d = dataset[1]
	print(d)

	frac = int(len(dataset)*0.9)

	train_dataset = dataset[:frac]
	test_dataset = dataset[frac:]
	print(f'Number of training graphs: {len(train_dataset)}')
	print(f'Number of test graphs: {len(test_dataset)}')
	logging.info(f'Number of training graphs: {len(train_dataset)}')
	logging.info(f'Number of test graphs: {len(test_dataset)}')

	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	d = train_dataset[0]
	print(d)
	print(d.face.T)
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	device = torch.device('cuda', args.device_idx)



	print(d.num_features)
	#model = Net(d.num_features, d.num_nodes, spiral_indices).to(device)
	if (args.pointnet):
		model = pointNet()
	else:
		model = Net(d.num_features, num_classes=1, spiral_length = args.seq_length)
	# model = model.cuda()
	# if torch.cuda.device_count() > 1:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# 	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	# 	model = nn.DataParallel(model, device_ids=[0, 1])#.cuda()
		# model = nn.parallel.DistributedDataParallel(model, device_ids=[0, 1,2,3])#.cuda()
		# spiral_indices = preprocess_spiral(d.face.T, args.seq_length).to(device)
	# # target = d.y_node.to(device)
	# print(model.device_ids)
	if (args.resume):
		try:
			model.load_state_dict(torch.load(f'model/ships_model_{args.samples}_length_{args.seq_length}_dilation_{args.dilation}_epochs_{args.epochs}_exp{args.exp}.pt'))
			logging.info("Saved model loaded")
			print("Saved model loaded")

			if (args.pretrained):
				newmodel = torch.nn.Sequential(*(list(model.children())))
				model = customModel(newmodel)
				print("pretrained model loaded")
				logging.info("pretrained model loaded")

		except:
			logging.info("Saved model not found")
			print("Saved model not found")
	logging.info(model)
	model.to(device)
	# model.to(f'cuda:{model.device_ids[0]}')


	optimizer = optim.Adam(model.parameters(),
					   lr=args.lr,
					   weight_decay=args.weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer,
									  args.decay_step,
									  gamma=args.lr_decay)


	# if  Train:
	run(model, train_loader, test_loader, d.num_nodes, args.epochs, optimizer, scheduler, device)

	torch.save(model.state_dict(), f'model/ships_model_{args.samples}_length_{args.seq_length}_dilation_{args.dilation}_epochs_{args.epochs}_exp{args.exp}.pt')
