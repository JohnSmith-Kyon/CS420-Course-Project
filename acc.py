from PIL import Image
import numpy as np
import os

def compute_acc(path_img, path_label):
    img = Image.open(path_img)
    label = Image.open(path_label)
    # Convert both images (RGB mode and L mode) into 1 mode.
    img = img.convert('1')
    label = label.convert('1')
    img = np.array(img)
    label = np.array(label)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if(img[i][j] == label[i][j]):
                if(label[i][j] == 0):
                    TN = TN + 1
                else:
                    TP = TP + 1
            else:
                if(label[i][j] == 0):
                    FP = FP + 1
                else:
                    FN = FN + 1
    return TN, TP, FP, FN

if __name__ == '__main__':
    acc = 0
    all_TN = 0
    all_TP = 0
    all_FP = 0
    all_FN = 0
    result_path = 'new_data\\result\\result_epoch_50\\'
    ref_path = 'new_data\\ref\\test_label\\'

    for i in range(5):
        i = str(i)
        img_file = os.path.join(result_path+ i + '_predict.png')
        label_file = os.path.join(ref_path+ i + '.png')
        TN, TP, FP, FN = compute_acc(img_file, label_file)
        all_TN = all_TN + TN
        all_TP = all_TP + TP
        all_FP = all_FP + FP
        all_FN = all_FN + FN

    acc = (all_TP + all_TN)/ (all_TP + all_FN + all_FP + all_TN)

    print(all_TP)
    print(all_FP)
    print(all_FN)
    print(all_TN)

    print(acc)










