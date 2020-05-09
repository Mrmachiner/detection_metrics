import argparse
import glob
import os
import shutil
import sys

import math
import _init_paths
from utils import BBFormat, CoordinatesType
from Evaluator import *
from config_para import config_metric
from Validate import validateFormats, validateMandatoryArgs, validateImageSize, validateCoordinatesTypes, validatePaths
from read_file import read_txt, read_json
def getBoundingBoxes(directory,
                     typefolder,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    if(typefolder == '.txt'):
        # Read ground truths
        return read_txt(directory,
                        isGT,
                        bbFormat,
                        coordType,
                        allBoundingBoxes,
                        allClasses,
                        imgSize)
    elif(typefolder == '.json'):
        return read_json(directory,
                        isGT,
                        bbFormat,
                        coordType,
                        allBoundingBoxes,
                        allClasses,
                        imgSize)


# Get current path to set default folders

currentPath = os.path.dirname(os.path.abspath(__file__))
args = config_metric()
iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = validateFormats(args.gtFormat, '-gtformat', errors)
detFormat = validateFormats(args.detFormat, '-detformat', errors)
# Groundtruth folder
if validateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
    gtFolder = validatePaths(args.gtFolder, '-gt/--gtfolder', errors)
else:
    # errors.pop()
    gtFolder = os.path.join(currentPath, 'groundtruths')
    if os.path.isdir(gtFolder) is False:
        errors.append('folder %s not found' % gtFolder)
# Coordinates types
gtCoordType = validateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = validateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = validateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = validateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
# Detection folder
if validateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
    detFolder = validatePaths(args.detFolder, '-det/--detfolder', errors)
else:
    # errors.pop()
    detFolder = os.path.join(currentPath, 'detections')
    if os.path.isdir(detFolder) is False:
        errors.append('folder %s not found' % detFolder)
if args.savePath is not None:
    savePath = validatePaths(args.savePath, '-sp/--savepath', errors)
else:
    savePath = os.path.join(currentPath, 'results')
# Validate savePath
# If error, show error messages
if len(errors) != 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Create directory to save results
shutil.rmtree(savePath, ignore_errors=True)  # Clear folder
os.makedirs(savePath)
# Show plot during execution
showPlot = args.showPlot

# print('iouThreshold= %f' % iouThreshold)
# print('savePath = %s' % savePath)
# print('gtFormat = %s' % gtFormat)
# print('detFormat = %s' % detFormat)
# print('gtFolder = %s' % gtFolder)
# print('detFolder = %s' % detFolder)
# print('gtCoordType = %s' % gtCoordType)
# print('detCoordType = %s' % detCoordType)
# print('showPlot %s' % showPlot)

# Get groundtruth boxes
allBoundingBoxes, allClasses, num_img = getBoundingBoxes(
    gtFolder, args.typefolder , True, gtFormat, gtCoordType, imgSize=imgSize)

num_annos = len(allBoundingBoxes._boundingBoxes)
# Get detected boxes
allBoundingBoxes, allClasses, num_img = getBoundingBoxes(
    detFolder, args.typefolder , False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
allClasses.sort()

evaluator = Evaluator()
acc_AP = 0
validClasses = 0

# Plot Precision x Recall curve
detections = evaluator.PlotPrecisionRecallCurve(
    allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=iouThreshold,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation, # EveryPointInterpolation ElevenPointInterpolation
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=True,  # Don't plot the interpolated precision curve
    savePath=savePath,
    showGraphic=showPlot)
# f = open(os.path.join(savePath, 'results11.txt'), 'w')
# f.write('Object Detection Metrics\adding folders with relative coordinates \n')
# f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
# f.write('Average Precision (AP), Precision and Recall per class:')

# # each detection is a class
for metricsPerClass in detections:

    # Get metric values per each class
    cl = metricsPerClass['class']
    ap = metricsPerClass['AP']
    precision = metricsPerClass['precision']
    recall = metricsPerClass['recall']
    totalPositives = metricsPerClass['total positives']
    total_TP = metricsPerClass['total TP']
    total_FP = metricsPerClass['total FP']

    print("FPPI: ", total_FP / num_img)

    print("MR: ", 100 * (1-(total_TP / num_annos)))

    if totalPositives > 0:
        validClasses = validClasses + 1
        acc_AP = acc_AP + ap
        prec = ['%.2f' % p for p in precision]
        rec = ['%.2f' % r for r in recall]
        ap_str = "{0:.2f}%".format(ap * 100)
        # ap_str = "{0:.4f}%".format(ap * 100)
        print('AP: %s (%s)' % (ap_str, cl))
        # f.write('\n\nClass: %s' % cl)
        # f.write('\nAP: %s' % ap_str)
        # f.write('\nPrecision: %s' % prec)
        # f.write('\nRecall: %s' % rec)

mAP = acc_AP / validClasses
mAP_str = "{0:.2f}%".format(mAP * 100)
print('mAP: %s' % mAP_str)
#f.write('\n\n\nmAP: %s' % mAP_str)
