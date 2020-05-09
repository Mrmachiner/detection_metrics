import os
import json
def is_an_score_file(filename):
    IMAGE_EXTENSIONS = ['.txt', '.csv', '.json']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return ext
    return False

def list_image_files(directory):
    files = sorted(os.listdir(directory))
    lst_path = []
    lst_type_path = []
    for f in files:
        if is_an_score_file(f) != False:
            path = os.path.join(directory, f)
            lst_type_path.append(is_an_score_file(f))
            lst_path.append(path)
    return lst_path, lst_type_path

def read_txt(path):
    list_dict = []
    with open(path) as f:
        # read = json.load(f)
        # print(data)
        read  = f.readlines()
        for line in read:
            score = 1
            bbox = []
            line = line.rstrip("\n\r").split(' ')
            if len(line) == 6:
                score = float(line[5])
            # print('line {}'.format(line))
            img_id = line[0]
            # for point in line[1:5]:
            #     bbox.append(int(float(point)))
            bbox = [int(float(point)) for point in line[1:5]]
            
            det_dict = {'image_id': img_id,
                        'category_id': 1,
                        'bbox': bbox,
                        'score': score,
                        'status': False}
            #print(det_dict)
            list_dict.append(det_dict)
    return list_dict

def read_json(path):
    list_dict = []
    with open(path) as f:
        read = json.load(f)
        # print(data)
        # read  = f.readlines()
            #path file
        if len(read) == 3:
            # file GT
            for line in read['objects']:
                bbox = []
                img_label = line['label']
                bbox.append(line['bbox']['x'])
                bbox.append(line['bbox']['y'])
                bbox.append(line['bbox']['w'])
                bbox.append(line['bbox']['h'])
                det_dict = {'img_label': img_label,
                            'bbox': bbox,}
                list_dict.append(det_dict)
        else:
            # file detect
            for line in read:
                bbox = []
                img_id = line['image_id']
                bbox.append(line['bbox']['x'])
                bbox.append(line['bbox']['y'])
                bbox.append(line['bbox']['w'])
                bbox.append(line['bbox']['h'])
                score = line['score']
                det_dict = {'image_id': img_id,
                            'category_id': 1,
                            'bbox': bbox,
                            'score': score}
                list_dict.append(det_dict)
    return list_dict

def read_file(path, type_path):
    if type_path == '.txt':
        lst_dict = read_txt(path)
        return lst_dict
    elif type_path == '.json':
        lst_dict = read_json(path)
        return lst_dict


def show_value_file(lst_value):
    for v in lst_value:
        print(v)

def load_images(path, n_file):
    """
    path : path file have two folder GT and Pre
    n_file : number file
    GT : Ground Truth
    Pre: Predict

    return: list value ground truth, predict
    """
    if n_file < 0:
        n_file = float("inf")
    GT_paths, Pre_paths = os.path.join(path, 'GT'), os.path.join(path, 'Pre')

    all_GT_paths, gt_type_path =  list_image_files(GT_paths)
    all_Pre_paths, pre_type_path = list_image_files(Pre_paths)
    
    value_GTs = []
    value_Pres = []

    for path_gt, gt_type, path_pre, pre_type in zip(all_GT_paths,gt_type_path, all_Pre_paths, pre_type_path):
        gt_value = read_file(path_gt, gt_type)
        pre_value = read_file(path_pre, pre_type)
        #print(value)
        value_GTs.append(gt_value)
        value_Pres.append(pre_value)
        #show_value_file(gt_value)
        if len(value_GTs) > n_file -1: break
    return value_GTs, value_Pres
    
def write_file(path, lst_dict):
    f = open(path, "w")
    for i in lst_dict:
        id = i['image_id']
        bbox = i['bbox']
        bbox[2] += bbox[0]
        bbox[3] = bbox[1]
        score = i['score']
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        l = str(id)+" "+ str(score)+" "+str(x1)+" " + str(y1)+" " + str(x2)+" " + str(y2) + "\n"
        f.writelines(l)
if __name__ == "__main__":
    value_GTs, value_Pres = load_images("../measure/Score", 5)
    #write_file("abcd1.txt", value_Pres[0])
    print("ok")
    #show_value_file(value_GTs)