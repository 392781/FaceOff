import adversarial_face_recognition as att
import glob
from tqdm import tqdm

"""
    Documentation for what is going on will come later

    For now this just takes 5 input IDs and 5 target IDs and then creates attacks for
    each one
"""

input_path = glob.glob('./faces/input/*.*')
target_path = glob.glob('./faces/target/*.*')
input_test_path = glob.glob('./faces/input_tests/*.*')
target_test_path = glob.glob('./faces/target_tests/*.*')

input_path.sort()
target_path.sort()
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
for face, _ in tqdm(input_list):
    mask_list.append(att.create_mask(face))

print('\nMASK OFFSET ----')
for i in tqdm(range(len(mask_list))):
    mask_list[i] = att.mask_offset(input_list[i], mask_list[i])
    
attack = att.Attack(input_list, target_list, mask_list, 'adamax')
attack.train(epochs = 45)

for image_path in input_test_path:
    input_test_list.append(att.detect_face(image_path)[0])

for image_path in target_test_path:
    target_test_list.append(att.detect_face(image_path)[0])

attack.results(input_test_list, target_test_list)