from __future__ import absolute_import, division, print_function

from copy import deepcopy

from utils import load_images

import numpy as np
import math
import os
import collections
import sys
import itertools
from operator import itemgetter
def pointHW_swap_2point(pointHW_detec, pointHW_gt):
	assert len(pointHW_detec) == 4
	assert len(pointHW_gt) == 4
	if pointHW_detec[0] < 0 or pointHW_detec[1] < 0 or pointHW_detec[2] < 0 or pointHW_detec[3] < 0:
		return False
	pointHW_detec[2] += pointHW_detec[0]
	pointHW_detec[3] = pointHW_detec[1]

	pointHW_gt[2] += pointHW_gt[0]
	pointHW_gt[3] = pointHW_gt[1]
	return pointHW_detec, pointHW_gt

def take_lst_box_imageID(lst_box, image_id):
	if image_id == 67:
		x = 0
	box_image_id = [dic['bbox'] for dic in lst_box if dic['image_id']==image_id]
	#print(box_image_id_1)
	return box_image_id


def single_detection_iou_multi_ground_truth(detection_rs, lst_ground_truth, image_id, iou_thre):
	"""
	Args:
		detection_rs : box [xmin, ymin, xmax, ymax]
		lst_ground_truth (dict) : dict = {'image_id': img_id,
										'category_id': 1,
										'bbox': bbox,
										'score': score}
		iou_thre : IoU threshold
		status_box_gt: array check lst_ground match
	Returns:
		true_positives 	:0 	score IoU detection vs GT > threhold
		false_positives :1	detection yes, ground_truth no
		true_negatives	:2	detection no , ground_truth no
		false_negatives :3	score IoU detection vs GT < threhold
		True : score IoU detection vs GT > threhold
		False :
	"""
	false_positives = 0
	true_positives 	= 1
		
	true_negatives	= 2
	false_negatives = 3	
	count = 0
	iouMax = sys.float_info.min
	leng = len(lst_ground_truth)
	detec = True
	local = -1
	for i in range(len(lst_ground_truth)):
		if lst_ground_truth[i]['image_id'] == image_id:
			if detec == False:
				return false_positives
			else:
				#box_detec, box_gt = pointHW_swap_2point(detection_rs, lst_ground_truth[i])
				_iou = iou(detection_rs['bbox'], lst_ground_truth[i]['bbox'])
				if _iou > iouMax:
					iouMax = _iou
					local = i
	if iouMax > iou_thre and lst_ground_truth[local]['status'] != True:
		lst_ground_truth[local]['status'] = True
		return true_positives, lst_ground_truth
	elif iouMax > iou_thre and lst_ground_truth[local]['status'] == True:
		return false_positives, lst_ground_truth
	else:
		return false_positives, lst_ground_truth


def match_detections(detection_results, lst_key_pre, ground_truth, lst_key_gt, overlap_threshold):
	""" Match detection results with gound truth and return true and false positive rates.
	This function will return a list of values as the true and false positive rates.
	These values represent the rates at increasing confidence thresholds.
	Args:
		detection_results (dict): Detection objects per image  
						dict = {'image_id': img_id,
								'category_id': 1,
								'bbox': bbox,
								'score': score}
		ground_truth (dict): Annotation objects per image
		overlap_threshold (Number): Minimum overlap threshold for true positive
	Returns:
		list: **[true_positives]**, **[false_positives]**, **num_annotations**
	"""
	t_positives = []
	f_positives = []
	f_negatives = []
	negatives = []
	lst_TP = []
	lst_FP = []		
	count_box = 0
	for i in range(len(lst_key_gt)):
		for j in range(len(lst_key_gt[i])):
			id_image = lst_key_gt[i][j]
			lstbox_dt = detection_results[i][id_image]
			lstbox_gt = ground_truth[i][id_image]
			leng = 0
			if len(ground_truth[i][id_image]) > len(detection_results[i][id_image]):
				leng = detection_results
			else:
				leng = len(detection_results[i][id_image])
			TP = np.zeros(len(ground_truth[i][id_image]))
			FP = np.zeros(len(ground_truth[i][id_image]))
			for k in range(len(lstbox_dt)):
				good, lstbox_gt = single_detection_iou_multi_ground_truth(lstbox_dt[k], lstbox_gt, id_image, overlap_threshold)
				if good == 1:
					TP[k] = 1
					t_positives.append(lstbox_dt[k])
				elif good == 0:
					FP[k] = 1
					f_positives.append(lstbox_dt[k])
				elif good == 3:
					f_negatives.append(lstbox_dt[k])
			lst_TP.append(TP)
			lst_FP.append(FP)
		# print("abcd TP",TP)
		# print("xyad FP",FP)
		# lst_TP.append(TP)
		# lst_FP.append(FP)
	print(lst_FP)
	print(lst_TP)
	tp = len(t_positives)
	fp = len(f_positives)
	fp += count_box - tp - fp
	return tp, fp

def match_detection_to_annotations(detection, annotations, overlap_threshold, overlap_fn):
	""" Compute the best match (largest overlap area) between a given detection and a list of annotations.
	Args:
		detection (brambox.boxes.detections.Detection): Detection to match
		annotations (list): Annotations to search for the best match
		overlap_threshold (Number): Minimum overlap threshold to consider detection and annotation as matched
		overlap_fn (function): Overlap area calculation function
	"""
	best_overlap = overlap_threshold
	best_annotation = None
	for i, annotation in enumerate(annotations):
		if annotation.clnum_annotations:
			overlap = overlap_fn(annotation, detection)
		if overlap < best_overlap:
			continue
		best_overlap = overlap
		best_annotation = i

	return best_annotation

def intersection(a, b):
	""" Calculate the intersection area between two boxes.
	Args:
		a (brambox.boxes.box.Box): First bounding box
		b (brambox.boxes.box.Box): Second bounding box
	Returns:
		Number: Intersection area
	"""
	intersection_top_left_x = max(a[0], b[0])
	intersection_top_left_y = max(a[1], b[1])
	intersection_bottom_right_x = min(a[2], b[2])
	intersection_bottom_right_y = min(a[3], b[3])

	intersection_width = intersection_bottom_right_x - intersection_top_left_x 
	intersection_height = intersection_bottom_right_y - intersection_top_left_y 
	
	if intersection_width <= 0 or intersection_height <= 0:
		return 0.0

	inter_area = intersection_width * intersection_height 

	return inter_area

def ioa(pred_box, gt_box, denominator='b'):
	""" Compute the intersection over area between two boxes a and b.
	The function returns the ``IOA``, which is defined as:
	:math:`IOA = \\frac { {intersection}(a, b) } { {area}(denominator) }`
	Args:
		a [xmin, ymin, xmax, ymax]: First bounding box
		b [xmin, ymin, xmax, ymax]: Second bounding box
		denominator (string, optional): String indicating from which box to compute the area; Default **'b'**
	Returns:
		Number: Intersection over union
	Note:
		The `denominator` can be one of 4 different values.
		If the parameter is equal to **'a'** or **'b'**, the area of that box will be used as the denominator.
		If the parameter is equal to **'min'**, the smallest of both boxes will be used
		and if it is equal to **'max'**, the biggest box will be used.
		If denominator something IoA is IoU, IoU = Area of Overlap/ Area of union
	"""
	x1_a, y1_a, x2_a, y2_a = pred_box
	x1_b, y1_b, x2_b, y2_b = gt_box

	if (x1_b > x2_b) or (y1_b > y2_b):
		raise AssertionError(
			"Second bounding box is malformed? Second box: {}".format(gt_box))
	if (x1_a > x2_a) or (y1_a > y2_a):
		raise AssertionError(
			"First bounding box is malformed? First box: {}".format(pred_box))

	a_width, a_height = x2_a - x1_a, y2_a - y1_a
	b_width, b_height = x2_b - x1_b, y2_b - y1_b

	inter_area = intersection(pred_box, gt_box)

	if denominator == 'min':
		div = min(a_width * a_height, b_width * b_height)
	elif denominator == 'max':
		div = max(a_width * a_height, b_width * b_height)
	elif denominator == 'a':
		div = a_width * a_height
	else:
		div = b_width * b_height
	return inter_area / div

def iou(pred_box, gt_box):
	"""Calculate IoU of single predicted and ground truth box

	Args:
		pred_box (list of floats): location of predicted object as
			[xmin, ymin, xmax, ymax]
		gt_box (list of floats): location of ground truth object as
			[xmin, ymin, xmax, ymax]

	Returns:
		float: value of the IoU for the two boxes.

	Raises:
		AssertionError: if the box is obviously malformed
	"""
	#print(pred_box[0] , pred_box[2], pred_box[1] ,pred_box[3])
	if (pred_box[0] > pred_box[2]) or (pred_box[1] > pred_box[3]):
		raise AssertionError(
			"Ground Truth box is malformed? true box: {}".format(pred_box))
	if (gt_box[0] > gt_box[2]) or (gt_box[1] > gt_box[3]):
		raise AssertionError(
			"Prediction box is malformed? pred box: {}".format(gt_box))


	if (pred_box[2] < gt_box[0] or gt_box[2] < pred_box[0] or pred_box[3] < gt_box[1] or gt_box[3] < pred_box[1]):
		return 0.0
	inter_area = intersection(pred_box, gt_box)
	pred_boxArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
	gt_boxArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
	combination_are = float(pred_boxArea + gt_boxArea - inter_area)
	if inter_area == combination_are:
		return 100
	_iou = inter_area / combination_are
	return _iou

def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
	"""Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

	Args:
		gt_boxes (list of list of floats): list of locations of ground truth
			objects as [xmin, ymin, xmax, ymax]
		pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
			and 'scores'
		iou_thr (float): value of IoU to consider as threshold for a
			true prediction.

	Returns:
		dict: true positives (int), false positives (int), false negatives (int)
	"""
	all_pred_indices = range(len(pred_boxes))
	all_gt_indices = range(len(gt_boxes))
	if len(all_pred_indices) == 0:
		tp = 0
		fp = 0
		fn = len(gt_boxes)
		return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
	if len(all_gt_indices) == 0:
		tp = 0
		fp = len(pred_boxes)
		fn = 0
		return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

	gt_idx_thr = []
	pred_idx_thr = []
	ious = []
	for ipb, pred_box in enumerate(pred_boxes):
		for igb, gt_box in enumerate(gt_boxes):
			_iou = iou(pred_box, gt_box)
			if _iou > iou_thr:
				gt_idx_thr.append(igb)
				pred_idx_thr.append(ipb)
				ious.append(_iou)

	args_desc = np.argsort(ious)[::-1]
	if len(args_desc) == 0:
		# No matches
		tp = 0
		fp = len(pred_boxes)
		fn = len(gt_boxes)
	else:
		gt_match_idx = []
		pred_match_idx = []
		for idx in args_desc:
			gt_idx = gt_idx_thr[idx]
			pr_idx = pred_idx_thr[idx]
			# If the boxes are unmatched, add them to matches
			if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
				gt_match_idx.append(gt_idx)
				pred_match_idx.append(pr_idx)
		tp = len(gt_match_idx)
		fp = len(pred_boxes) - len(pred_match_idx)
		fn = len(gt_boxes) - len(gt_match_idx)

	return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

def calc_precision_recall(img_results):
	"""Calculates precision and recall from the set of images

	Args:
		img_results (dict): dictionary formatted like:
			{
				'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
				'img_id2': ...
				...
			}

	Returns:
		tuple: of floats of (precision, recall)
	"""
	true_pos = 0; false_pos = 0; false_neg = 0
	##a = img_results.items()
	##xx = img_results['true_pos']
	for i in range(len(img_results)):
		true_pos += img_results[i]['true_pos']
		false_pos += img_results[i]['false_pos']
		false_neg += img_results[i]['false_neg']

	try:
		precision = true_pos/(true_pos + false_pos)
	except ZeroDivisionError:
		precision = 0.0
	try:
		recall = true_pos/(true_pos + false_neg)
	except ZeroDivisionError:
		recall = 0.0

	return (precision, recall)

def get_model_scores_map(pred_boxes):
	"""Creates a dictionary of from model_scores to image ids.

	Args:
		pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

	Returns:
		dict: keys are model_scores and values are image ids (usually filenames)

	"""
	model_scores_map = {}
	for img_id, val in pred_boxes.items():
		if len(val)>0:
			for score in val['scores']:
				if score not in model_scores_map.keys():
					model_scores_map[score] = [img_id]
				else:
					model_scores_map[score].append(img_id)
	return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5):
	"""Calculates average precision at given IoU threshold.

	Args:
		gt_boxes (list of list of floats): list of locations of ground truth
			objects as [xmin, ymin, xmax, ymax]
		pred_boxes (list of list of floats): list of locations of predicted
			objects as [xmin, ymin, xmax, ymax]
		iou_thr (float): value of IoU to consider as threshold for a
			true prediction.

	Returns:
		dict: avg precision as well as summary info about the PR curve

		Keys:
			'avg_prec' (float): average precision for this IoU threshold
			'precisions' (list of floats): precision value for the given
				model_threshold
			'recall' (list of floats): recall value for given
				model_threshold
			'models_thrs' (list of floats): model threshold value that
				precision and recall were computed for.
	"""
	model_scores_map = get_model_scores_map(pred_boxes)
	sorted_model_scores = sorted(model_scores_map.keys())

	# Sort the predicted boxes in descending order (lowest scoring boxes first):
	for img_id in pred_boxes.keys():
		if len(pred_boxes[img_id]) > 0:
			arg_sort = np.argsort(pred_boxes[img_id]['scores'])
			pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
			pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

	pred_boxes_pruned = deepcopy(pred_boxes)

	precisions = []
	recalls = []
	model_thrs = []
	img_results = {}
	# Loop over model score thresholds and calculate precision, recall
	for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
		# On first iteration, define img_results for the first time:
		img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
		for img_id in img_ids:
			gt_boxes_img = gt_boxes[img_id]
			if len(pred_boxes_pruned[img_id]) > 0:
				box_scores = pred_boxes_pruned[img_id]['scores']
				start_idx = 0
				for score in box_scores:
					if score < model_score_thr:
                         # pred_boxes_pruned[img_id]
						start_idx += 1
					else:
						break

				# Remove boxes, scores of lower than threshold scores:
				pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
				pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

				# Recalculate image results for this image
				img_results[img_id] = get_single_image_results(
					gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)
			else:
				img_results[img_id] = {'true_pos': 0, 'false_pos': 0, 'false_neg': len(gt_boxes_img)}
		prec, rec = calc_precision_recall(img_results)
		precisions.append(prec)
		recalls.append(rec)
		model_thrs.append(model_score_thr)

	precisions = np.array(precisions)
	recalls = np.array(recalls)
	prec_at_rec = []
	for recall_level in np.linspace(0.0, 1.0, 11):
		try:
			args = np.argwhere(recalls >= recall_level).flatten()
			prec = max(precisions[args])
		except ValueError:
			prec = 0.0
		prec_at_rec.append(prec)
	avg_prec = np.mean(prec_at_rec)

	return {
		'avg_prec': avg_prec,
		'precisions': precisions,
		'recalls': recalls,
		'model_thrs': model_thrs}

def mr_fppi(detections, ground_truth, overlap_threshold=0.5):
	""" Compute a list of miss-rate, FPPI and log-average missing rate values
		that can be plotted into a graph.
	Args:
		detections (dict): Detection objects per image
		ground_truth (dict): Annotation objects per image
		overlap_threshold (Number, optional): Minimum iou threshold for true positive; Default **0.5**
	Returns:
		tuple: **[miss-rate_values]**, **[fppi_values]**, **avg_log_miss_rate**
	"""
	num_images = len(ground_truth)
	tps, fps, num_annotations = match_detections(detections, ground_truth, overlap_threshold)

	miss_rate = []
	log_miss_rate = []
	fppi = []
	for tp, fp in zip(tps, fps):
		miss_rate.append(math.log(1 - (tp / num_annotations), 10))
		log_miss_rate.append(1 - (tp/num_annotations))
		fppi.append(fp / num_images)

	avg_log_miss_rate = np.mean(log_miss_rate)

	return miss_rate, fppi, avg_log_miss_rate

def benchmarks(tp, fp):
	precision = tp / ( tp + fp)
	fppi = fp / ( tp + fp)
	mr = math.log(fppi, 10) + 1
	print("Precision: %f, FPPI: %f, MR2: %f" % (precision, fppi, mr))

def group_value_id(lst_value):
	"""
	Args:
		detection_relst_value (dict): Detection objects per image  
						dict = {'image_id': img_id,
								'category_id': 1,
								'bbox': bbox,
								'score': score}
	return group box by ID image 
	[{[id:1], [id:1], ...}, {dict}]
	"""
	lst_gr_value_file = []
	lst_gr_key_file = []
	for lst in lst_value:
		sorted_image_id = sorted(lst, key=itemgetter('image_id'))
		lst_group_gt = {}
		lst_key = []
		for key, group in itertools.groupby(sorted_image_id, key=lambda x:x['image_id']):
			#print (key)
			lst_key.append(key)
			temp =list(group)
			lst_group_gt[key] = temp
		lst_gr_value_file.append(lst_group_gt)
		lst_gr_key_file.append(lst_key)
	return lst_gr_value_file, lst_gr_key_file

if __name__ == "__main__":

	#iou, ioa, get_single_image_results, calc_precision_recall --------> ok

	path_value = "../measure/Score"
	lst_value_gt, lst_value_pre = load_images(path_value,4)
	print("Box ground truth: %i "%len(lst_value_gt[0]))
	print("Box Predict: %i" % len(lst_value_pre[0]))
	pred_box = [39, 63, 203, 112]
	gt_box = [54, 66, 198, 114]

	lst_pred_box = []
	lst_gt_box = []

	lst_pred_box.append(pred_box)
	lst_gt_box.append(gt_box)
	gr_id_gt, lst_key_gt = group_value_id(lst_value_gt)
	gr_id_pre, lst_key_pre = group_value_id(lst_value_pre)
	
	tp, fp = match_detections(gr_id_pre, lst_key_pre, gr_id_gt, lst_key_gt, 0.5 )
	total = tp + fp
	benchmarks(tp, fp)
	print("tp: %i, fp: %i" % (tp, fp))
	#print("Accuracy	", (tp+tn)/total)


	#gapai = get_avg_precision_at_iou(gt_box, pred_box)
	#print(gapai)
	
 	
