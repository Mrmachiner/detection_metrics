import os
import glob
import json
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import *
def read_txt(directory,
            isGT,
            bbFormat,
            coordType,
            allBoundingBoxes=None,
            allClasses=None,
            imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses

def default_value(nameOfImage, idClass, coordType, imgSize, isGT, bbFormat):
    idClass = '0'  # class
    x = 0 #x_topleft
    y = 0 #y_topleft
    w = 0
    h = 0
    confidence = 0
    if isGT:
        bb = BoundingBox(
            nameOfImage,
            idClass,
            x,
            y,
            w,
            h,
            coordType,
            imgSize,
            BBType.GroundTruth,
            format=bbFormat)
    else:
        bb = BoundingBox(
            nameOfImage,
            idClass,
            x,
            y,
            w,
            h,
            coordType,
            imgSize,
            BBType.Detected,
            confidence,
            format=bbFormat)
    return bb

def read_json(directory,
            isGT,
            bbFormat,
            coordType,
            allBoundingBoxes=None,
            allClasses=None,
            imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    os.chdir(directory)
    files = glob.glob("*.json")
    num_img = len(files)
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, himport jsoneight" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    i = 0
    for f in files:
        nameOfImage = f.replace(".json", "")
        path_file = os.path.join(directory, f)
        with open(path_file) as f:
            fh1 = json.load(f)
            for line in fh1['objects']:
                # line = line.replace("\n", "")
                # if line.replace(' ', '') == '':
                #     continue
                # splitLine = line.split(" ")
                if len(line)!=3:
                    bb = default_value(nameOfImage, idClass, coordType,imgSize, isGT, bbFormat)
                    i += 1
                else:
                    if isGT:
                        if "-" in nameOfImage:
                            _ ,nameOfImage = nameOfImage.split("-")
                        # idClass = int(splitLine[0]) #class
                        idClass = line['label']  # class
                        x = float(line['bbox']['x_topleft']) #x_topleft
                        y = float(line['bbox']['y_topleft']) #y_topleft
                        w = float(line['bbox']['w'])
                        h = float(line['bbox']['h'])
                        bb = BoundingBox(
                            nameOfImage,
                            idClass,
                            x,
                            y,
                            w,
                            h,
                            coordType,
                            imgSize,
                            BBType.GroundTruth,
                            format=bbFormat)
                    else:
                        # idClass = int(splitLine[0]) #class
                        idClass = line['label']  # class
                        confidence = float(line['conf'])
                        x = float(line['bbox']['x_topleft'])
                        y = float(line['bbox']['y_topleft'])
                        w = float(line['bbox']['w'])
                        h = float(line['bbox']['h'])
                        bb = BoundingBox(
                            nameOfImage,
                            idClass,
                            x,
                            y,
                            w,
                            h,
                            coordType,
                            imgSize,
                            BBType.Detected,
                            confidence,
                            format=bbFormat)
                allBoundingBoxes.addBoundingBox(bb)
                if idClass not in allClasses:
                    allClasses.append(idClass)
        #fh1.close()
    return allBoundingBoxes, allClasses, num_img

