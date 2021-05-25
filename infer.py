from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 0.7

transform_img = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

def get_model():
	net = get_model_instance_segmentation(2)
	net.eval()
	net.load_state_dict(torch.load('model.pth'), strict=False)
	return net

def load_img(path):
	img = Image.open(path).convert('RGB')
	img_t = transform_img(img).unsqueeze(0)
	return img, img_t

def predict(imgs, imgs_t, net, save_image=False, threshold=THRESHOLD):
	outputs = net(imgs_t)
	predictions = []
	for i in range(len(outputs)):
		pred_proba = outputs[i]['scores'].detach().numpy()
		sub_predictions = []
		for j in range(len(pred_proba)):
			if pred_proba[j]<threshold:
				continue
			mask = outputs[i]['masks'][j].detach().numpy()[0]
			mask = np.round(mask).astype(int)
			coords = outputs[i]['boxes'][j]
			coords = np.array(coords.detach(), dtype=np.int32)
			img_c = np.array(imgs[i])
			img_c = img_c[coords[1]:coords[3],coords[0]:coords[2],:]
			mask = mask[coords[1]:coords[3],coords[0]:coords[2]]
			mask = np.expand_dims(mask, axis=2)
			img_c = np.array(img_c*mask, dtype=np.uint8)
			if save_image:
				plt.imsave(f'outputs/save_{i}_{j}.jpg', img_c)
			sub_predictions.append(img_c)
		predictions.append(sub_predictions)
	return predictions

def main():
	net = get_model()
	img, img_t = load_img(path='test/5.png')
	preds = predict([img], img_t, net, save_image=True)

if __name__ == '__main__':
	main()
