import argparse
import glob
import os
import shutil
def config_metric():
    #currentPath = os.path.dirname(os.path.abspath(__file__))
    VERSION = '0.1 (beta)'
    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description='This project applies the most popular metrics used to evaluate object detection '
        'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
        'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    # formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Positional arguments
    # Mandatory
    parser.add_argument(
        '-gt',
        '--gtfolder',
        dest='gtFolder',
        default=os.path.join('/home/minhhoang/Downloads/kitti/pydriver/pydriver-master/out_json/'),
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-det',
        '--detfolder',
        dest='detFolder',
        default=os.path.join('/home/minhhoang/Desktop/Data_test/head_detec/tu/dt/mobile_val_predict_0.5/'),
        metavar='',
        help='folder containing your detected bounding boxes')
    parser.add_argument(
        '-typefolder',
        dest='typefolder',
        default='.json',
        help='format folder .txt or .json')
    # Optional
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '-gtformat',
        dest='gtFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the ground truth bounding boxes: '
        '(\'xywh\': <left> <top> <width> <height>)'
        ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-detformat',
        dest='detFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the detected bounding boxes '
        '(\'xywh\': <left> <top> <width> <height>) '
        'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-gtcoords',
        dest='gtCoordinates',
        default='abs',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
        'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-detcoords',
        default='abs',
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
        'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-imgsize',
        dest='imgSize',
        metavar='',
        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument(
        '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
    parser.add_argument(
        '-np',
        '--noplot',
        dest='showPlot',
        action='store_false',
        help='no plot is shown during execution')
    args = parser.parse_args()
    return args