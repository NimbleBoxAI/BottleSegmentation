from pycocotools.coco import COCO
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms 
from sklearn.model_selection import train_test_split
import torchvision
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_model_instance_segmentation(num_classes):
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
													   hidden_layer,
													   num_classes)
	return model

class Dataset(torch.utils.data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, img_id, coco):
		'Initialization'
		self.img_id = img_id
		self.coco = coco
		self.transform_img = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.img_id)

	def __getitem__(self, index):
		'Generates one sample of data'
		img = self.coco.loadImgs(self.img_id[index])[0]
		annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=44)
		anns = self.coco.loadAnns(annIds)
		img_file = img['file_name']
		image = Image.open(f'coco/train2017/{img_file}').convert('RGB')
		image = self.transform_img(image)
		masks = []
		for i in range(len(anns)):
			mask = np.zeros((img['height'],img['width']))
			mask = np.maximum(self.coco.annToMask(anns[i]), mask)
			masks.append(mask)
		masks = torch.as_tensor(masks, dtype=torch.uint8)
		boxes = []
		for i in range(len(anns)):
			pos = anns[i]['bbox']
			xmin = pos[0]
			xmax = xmin+pos[2]
			ymin = pos[1]
			ymax = ymin+pos[3]
			boxes.append([xmin, ymin, xmax, ymax])
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.ones((len(anns),), dtype=torch.int64)
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		return image, target


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	num_classes = 2
	annFile='coco/annotations_trainval2017/annotations/instances_train2017.json'
	coco=COCO(annFile)
	catIDs = coco.getCatIds()
	cats = coco.loadCats(catIDs)
	img_Ids = coco.getImgIds(catIds=44)
	train_imgIds, val_imgIds = train_test_split(img_Ids, test_size=0.1)
	train_dataset = Dataset(train_imgIds, coco)
	test_dataset = Dataset(val_imgIds, coco)
	data_loader = DataLoader(train_dataset, batch_size=4, num_workers=os.cpu_count(),
		collate_fn=utils.collate_fn)
	data_loader_test = DataLoader(test_dataset, batch_size=1, num_workers=os.cpu_count(),
		collate_fn=utils.collate_fn)

	model = get_model_instance_segmentation(num_classes)

	model.to(device)

	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
								momentum=0.9, weight_decay=0.0005)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												   step_size=3,
												   gamma=0.1)
	num_epochs = 10

	for epoch in range(num_epochs):
		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
		lr_scheduler.step()
		torch.save(model.state_dict(),"model.pth")
		#evaluate(model, data_loader_test, device, epoch)

if __name__ == '__main__':
	main()
