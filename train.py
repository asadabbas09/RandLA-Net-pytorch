import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from numpy import asarray
from numpy import save
import torch_geometric.transforms as T

# from data import data_loaders
from model import RandLANet
# from model_sigmoid import RandLANet
# from utils.tools import Config as cfg
# from utils.metrics import accuracy, intersection_over_union
import meshio
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import logging


os.makedirs('log', exist_ok=True)
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
				# pos = np.expand_dims(mesh_io.points.astype(np.float32), axis=0)

				# pos = torch.from_numpy(pos).to(torch.float)
				pos = torch.from_numpy(mesh_io.points.astype(np.float32)).to(torch.float)
				face = torch.from_numpy(mesh_io.cells[0].data.astype(np.float32)).to(torch.long).t().contiguous()

				# print(face.shape, pos.shape)

				f = mesh_io.cells[0].data#.astype(np.float32)
				#print(f.shape[1])
				# spiral = preprocess_spiral(f, args.seq_length, dilation=args.dilation)
				cx = drag_list_cx1[i]
				cy = drag_list_cy1[i]
				#print(spiral)


				# indices = spiral.to(torch.long).contiguous()


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


				data = Data(pos=pos, face=face, y_node = y_node, y_stress = y_stress, cx = cx, cy = cy)

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


def meshfile_list(path_dir):#'/media/asad/Work/Documents/Emulator/AustalDataset/extracted/'):
		keyword_p = 'vessel.vtk'
		keyword_d = 'drag_coeffs.json'
		mesh_list = []
		drag_list_cx = []
		drag_list_cy = []
		# print(path_dir)
		for root, dirs, files in os.walk(path_dir):
			# print(root, dirs, files)
			path = root.split(os.sep)
			for file in files:
				if keyword_p in file:
					mesh_list.append(root+'/'+file)
					# print(root+'/'+file, " has the keyword p")
					# mesh_io = meshio.read(root+'/'+file)
					# print(mesh_io)
					# y_node = mesh_io.point_data['p'].astype(np.float32)
					# ynode_list.append(y_node)

					# print(root+'/'+file, " has the keyword p")
				if keyword_d in file:
					with open(root+'/'+file) as json_file:
						data = json.load(json_file)
						drag_list_cx.append(data['cx'])
						drag_list_cy.append(data['cy'])
					# print(root+'/'+file, " has the keyword p")

		return mesh_list, drag_list_cx, drag_list_cy
# print(f"Looking for files in {args.dataset}")





def evaluate(model, loader, criterion, device):
	# model.eval()
	model.train()
	losses = []
	accuracies = []
	ious = []
	with torch.no_grad():
		for data in tqdm(loader, desc='Validation', leave=False):

			data = data.to(args.gpu)


			# points = data.x.float()#points.to(args.gpu)
			points = data.pos.float()#points.to(args.gpu)
			labels = data.y_node#labels.to(args.gpu)
			points = points.unsqueeze(0)
			# print('points',  np.shape(points))
			# print('labels',  np.shape(labels))
			# points = points.to(device)
			# labels = labels.to(device)

			scores = model(points)
			# scores = scores[0,0,:]
			loss = criterion(scores, labels)
			losses.append(loss.cpu().item())
			# accuracies.append(accuracy(scores, labels))
			# ious.append(intersection_over_union(scores, labels))
	return np.mean(losses), 0, 0# np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)


def train(args):
	# train_path = args.dataset / args.train_dir
	# val_path = args.dataset / args.val_dir
	logs_dir = args.logs_dir / args.name
	logs_dir.mkdir(exist_ok=True, parents=True)



	# # determine number of classes
	# try:
	# 	with open(args.dataset / 'classes.json') as f:
	# 		labels = json.load(f)
	# 		num_classes = len(labels.keys())
	# except FileNotFoundError:
	# 	num_classes = int(input("Number of distinct classes in the dataset: "))
	#
	# train_loader, val_loader = data_loaders(
	# 	args.dataset,
	# 	args.dataset_sampling,
	# 	batch_size=args.batch_size,
	# 	num_workers=args.num_workers,
	# 	pin_memory=True
	# )

	dataset = torch.load(f'datasets/ships/ships_samples_{args.samples}_cxy.pt')
	print(dataset)
	frac = int(len(dataset)*0.9)

	train_dataset = dataset[:frac]
	test_dataset = dataset[frac:]

	print(f'Number of training graphs: {len(train_dataset)}')
	print(f'Number of test graphs: {len(test_dataset)}')

	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	d = train_dataset[0]
	print(d)


	# d_in = next(iter(train_loader))[0].size(-1)

	model = RandLANet(
		3,
		1,
		num_neighbors=args.neighbors,
		decimation=args.decimation,
		device=args.gpu
	)
	print(model)
	# print('Computing weights...', end='\t')
	# samples_per_class = np.array(cfg.class_weights)
	#
	# n_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=args.gpu)
	# ratio_samples = n_samples / n_samples.sum()
	# weights = 1 / (ratio_samples + 0.02)
	#
	# print('Done.')
	# print('Weights:', weights)
	# criterion = nn.MSELoss()#nn.CrossEntropyLoss(weight=weights)
	criterion = nn.L1Loss()#nn.CrossEntropyLoss(weight=weights)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
	# optimizer = torch.optim.AdamW(model.parameters(), lr=args.adam_lr)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)

	first_epoch = 1
	if args.load:
		print(list((args.logs_dir / args.load).glob('*.pth')))
		path = max(list((args.logs_dir / args.load).glob('*.pth')))
		print(f'Loading {path}...')
		checkpoint = torch.load(path)
		first_epoch = checkpoint['epoch']+1
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		# print(checkpoint['scheduler_state_dict']['_last_lr'])
	optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr, amsgrad = False)
	with SummaryWriter(logs_dir) as writer:
		for epoch in range(first_epoch, args.epochs+1):
			print(f'=== EPOCH {epoch:d}/{args.epochs:d} ===')
			t0 = time.time()
			# Train
			model.train()

			# metrics
			losses = []
			accuracies = []
			ious = []

			# iterate over dataset
			for data in tqdm(train_loader, desc='Training', leave=False):

				data = data.to(args.gpu)


				# points = data.x.float()#points.to(args.gpu)
				points = data.pos.float()#points.to(args.gpu)
				# print(points)
				labels = data.y_node#labels.to(args.gpu)
				points = points.unsqueeze(0)
				# print('points',  np.shape(points))
				# print('labels',  np.shape(labels))
				optimizer.zero_grad()

				scores = model(points)

				# print('scores',  np.shape(scores[0,0,:]))
				# scores = scores[0,0,:]
				# print(data.y_node)
				# print(scores)
				# logp = torch.distributions.utils.probs_to_logits(scores, is_binary=False)
				loss = criterion(scores, labels)
				# logpy = torch.gather(logp, 1, labels)
				# loss = -(logpy).mean()
				# print(loss)
				loss.backward()

				optimizer.step()

				losses.append(loss.cpu().item())
				# accuracies.append(accuracy(scores, labels))
				# ious.append(intersection_over_union(scores, labels))

			scheduler.step()

			accs = 0#np.nanmean(np.array(accuracies), axis=0)
			ious = 0#np.nanmean(np.array(ious), axis=0)

			val_loss, val_accs, val_ious = evaluate(
				model,
				test_loader,
				criterion,
				args.gpu
			)

			loss_dict = {
				'Training loss':    np.mean(losses),
				'Validation loss':  val_loss
			}
			# acc_dicts = [
			# 	{
			# 		'Training accuracy': acc,
			# 		'Validation accuracy': val_acc
			# 	} for acc, val_acc in zip(accs, val_accs)
			# ]
			# iou_dicts = [
			# 	{
			# 		'Training accuracy': iou,
			# 		'Validation accuracy': val_iou
			# 	} for iou, val_iou in zip(ious, val_ious)
			# ]

			t1 = time.time()
			d = t1 - t0
			# Display results
			for k, v in loss_dict.items():
				print(f'{k}: {v:.7f}', end='\t')
				logging.info(f'{k}: {v:.7f}')
			print()

			# print('Accuracy     ', *[f'{i:>5d}' for i in range(num_classes)], '   OA', sep=' | ')
			# print('Training:    ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in accs], sep=' | ')
			# print('Validation:  ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in val_accs], sep=' | ')

			# print('IoU          ', *[f'{i:>5d}' for i in range(num_classes)], ' mIoU', sep=' | ')
			# print('Training:    ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in ious], sep=' | ')
			# print('Validation:  ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in val_ious], sep=' | ')

			print('Time elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)))

			# send results to tensorboard
			writer.add_scalars('Loss', loss_dict, epoch)

			# for i in range(num_classes):
			# 	writer.add_scalars(f'Per-class accuracy/{i+1:02d}', acc_dicts[i], epoch)
			# 	writer.add_scalars(f'Per-class IoU/{i+1:02d}', iou_dicts[i], epoch)
			# writer.add_scalars('Per-class accuracy/Overall', acc_dicts[-1], epoch)
			# writer.add_scalars('Per-class IoU/Mean IoU', iou_dicts[-1], epoch)

			if epoch % args.save_freq == 0:
				torch.save(
					dict(
						epoch=epoch,
						model_state_dict=model.state_dict(),
						optimizer_state_dict=optimizer.state_dict(),
						scheduler_state_dict=scheduler.state_dict()
					),
					args.logs_dir / args.name / f'checkpoint_{epoch:02d}.pth'
				)


if __name__ == '__main__':

	"""Parse program arguments"""
	parser = argparse.ArgumentParser(
		prog='RandLA-Net',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	base = parser.add_argument_group('Base options')
	expr = parser.add_argument_group('Experiment parameters')
	param = parser.add_argument_group('Hyperparameters')
	dirs = parser.add_argument_group('Storage directories')
	misc = parser.add_argument_group('Miscellaneous')

	base.add_argument('--dataset', type=Path, help='location of the dataset',
						default='/media/asad/Work/Documents/Emulator/AustalDataset/extracted/')

	expr.add_argument('--epochs', type=int, help='number of epochs',
						default=5000)
	expr.add_argument('--load', type=str, help='model to load',
						default='')

	param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
						default=1e-2)
	param.add_argument('--batch_size', type=int, help='batch size',
						default=1)
	param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
						default=4)
	param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
						default='active_learning', choices=['active_learning', 'naive'])
	param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
						default=16)
	param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
						default=0.99)

	dirs.add_argument('--test_dir', type=str, help='location of the test set in the dataset dir',
						default='test')
	dirs.add_argument('--train_dir', type=str, help='location of the training set in the dataset dir',
						default='train')
	dirs.add_argument('--val_dir', type=str, help='location of the validation set in the dataset dir',
						default='val')
	dirs.add_argument('--logs_dir', type=Path, help='path to tensorboard logs',
						default='runs')

	misc.add_argument('--gpu', type=int, help='which GPU to use (-1 for CPU)',
						default=0)
	misc.add_argument('--name', type=str, help='name of the experiment',
						default=None)
	misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
						default=0)
	misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
						default=10)

	parser.add_argument('--create_dataset',  action='store_true')
	parser.add_argument('--samples', type=int, default=10)

	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO, filename=f'log/ships_samples_{args.samples}_length_{args.adam_lr}_dilation_{args.decimation}_exp{args.neighbors}.log', filemode='w',format='%(message)s')

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




		# global ymean
		# ymean = []
		# global ystd
		# ystd = []
		# global ynode_list
		# ynode_list = []
		# for meshes in mesh_list1:
		# 	mesh_io = meshio.read(meshes)
		# 	y_node = mesh_io.point_data['p'].astype(np.float32)
		# 	ynode_list.append(y_node)
		#
		# print('ynode_list', np.shape(ynode_list))
		# ymean = np.mean(ynode_list)
		# ystd = np.std(ynode_list)
		#
		# print(np.shape(ymean))
		# print(np.shape(ystd))



		print(f"creating dataset for {args.samples}")
		print(mesh_list1)
		pre_transform = T.Compose([T.FaceToEdge(remove_faces=False), T.Constant(value=1)])

		# dataset = MyDataset(f'processdata/ships_{args.samples}_{args.seq_length}_{args.dilation}', pre_transform=T.FaceToEdge(remove_faces=False))
		# dataset = MyDataset(f'datasets/ships/processdata/ships_{args.samples}')
		dataset = MyDataset(f'datasets/ships/processdata/ships_{args.samples}', pre_transform = T.NormalizeScale())

		torch.save(dataset, f'datasets/ships/ships_samples_{args.samples}_cxy.pt')


	if args.gpu >= 0:
		if torch.cuda.is_available():
			args.gpu = torch.device(f'cuda:{args.gpu:d}')
		else:
			warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
			args.gpu = torch.device('cpu')
	else:
		args.gpu = torch.device('cpu')

	if args.name is None:
		if args.load:
			args.name = args.load
		else:
			args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

	t0 = time.time()
	train(args)
	t1 = time.time()

	d = t1 - t0
	print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
