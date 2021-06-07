from glob import glob
import os.path as osp
import os

orgin_label = 'labels/train2017'
label_list = glob(orgin_label + '/*.txt')
output_path = 'new_labels'
os.makedirs(output_path, exist_ok=True)

for single_label in label_list:
    with open(single_label, 'r') as f:
        f2 = open(osp.join(output_path, osp.basename(single_label)), 'w')
        lines = f.read().splitlines()
        for line in lines:
            line = line.split(' ')
            cls, x_center, y_center, width, height = \
                line[0], float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x1 = round(x_center - width / 2, 4)
            y1 = round(y_center - height / 2, 4)
            x2 = round(x_center + width / 2, 4)
            y2 = round(y_center - height / 2, 4)
            x3 = round(x_center + width / 2, 4)
            y3 = round(y_center + height / 2, 4)
            x4 = round(x_center - width / 2, 4)
            y4 = round(y_center + height / 2, 4)
            out_line = cls + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(x3) + ' ' + str(
                y3) + ' ' + str(x4) + ' ' + str(y4)
            f2.write(out_line + '\n')
        f2.close()
