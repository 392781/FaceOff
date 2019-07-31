import adversarial_face_recognition as att
import glob
from tqdm import tqdm

"""
    Documentation for what is going on will come later

    For now this just takes 5 input IDs and 5 target IDs and then creates attacks for
    each one
"""

input_path = glob.glob('./faces/input_tests/*.*')
target_path = glob.glob('./faces/target/*.*')
mask_path = glob.glob('./results/experiment_1/delta/*.*')
input_mask_path = glob.glob('./results/experiment_1/input/*.*')
input_test_path = glob.glob('./faces/input/*.*')
target_test_path = glob.glob('./faces/target_tests/*.*')

input_path.sort()
target_path.sort()
mask_path.sort()
input_mask_path.sort()
input_test_path.sort()
target_test_path.sort()

input_list = []
target_list = []
input_test_list = []
target_test_list = []
mask_list = []

print('\nINPUTS ---------')
for image_path in tqdm(input_path):
    input_list.append(att.detect_face(image_path))
    
print('\nTARGETS --------')
for image_path in tqdm(target_path):
    target_list.append(att.detect_face(image_path))

print('\nMASKS ----------')
for image_path in tqdm(mask_path):

    mask_list.append(att.fr.load_image_file(image_path))

print('\nMASK OFFSET ----')
for i in tqdm(range(len(mask_list))):
    coor = att.detect_face(input_mask_path[i])[1]
    mask_list[i] = att.mask_offset(input_list[i], mask_list[i], coor)
    
attack = att.Attack(input_list, target_list, mask_list, 'adamax')

for image_path in input_test_path:
    input_test_list.append(att.detect_face(image_path)[0])

for image_path in target_test_path:
    target_test_list.append(att.detect_face(image_path)[0])

attack.results(input_test_list, target_test_list, '/experiment_2/')