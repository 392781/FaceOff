import adversarial_face_recognition as att
import glob
from tqdm import tqdm

"""
    This experiment tests the transferability of masks for the same input IDs but
    different images

    RESULTS: (all euclidean distances)
    inputs vs ground truths
    0.6919291615486145
    0.710687518119812
    0.896115243434906
    0.45093584060668945
    0.6631213426589966

    inputs vs target
    0.719158411026001
    1.1436659097671509
    1.0781880617141724
    1.1135917901992798
    1.2560782432556152

    inputs vs target 2
    0.6480927467346191
    1.3156074285507202
    1.0370244979858398
    1.2272032499313354
    1.0253299474716187

    target vs target 2
    0.5311635136604309
    0.6634145379066467
    0.7484209537506104
    0.45897066593170166
    0.7643409371376038

    adversarial vs ground truths
    0.7142927646636963
    0.8391417860984802
    0.9934453964233398
    0.6173223257064819
    0.8968896269798279

    adversarial vs target
    0.9491204023361206
    0.9699550271034241
    1.0911160707473755
    1.118059754371643
    1.2404298782348633

    adversarial vs target 2
    0.9138741493225098
    1.235718846321106
    1.0918978452682495
    1.2216910123825073
    1.0198875665664673
"""

# Gather all image locations, this includes the created masks from the previous
# experiment that will be applied to new faces
input_path = glob.glob('./faces/input_tests/*.*')
target_path = glob.glob('./faces/target/*.*')
mask_path = glob.glob('./results/experiment_1/delta/*.*')
input_mask_path = glob.glob('./results/experiment_1/input/*.*')
input_test_path = glob.glob('./faces/input/*.*')
target_test_path = glob.glob('./faces/target_tests/*.*')

# Sort each image location by name so that each identity corresponds 
# for testing
input_path.sort()
target_path.sort()
mask_path.sort()
input_mask_path.sort()
input_test_path.sort()
target_test_path.sort()

# List initialization for the attack class inputs
input_list = []
target_list = []
input_test_list = []
target_test_list = []
mask_list = []

# Detect faces for all the image locations and apply masks for new faces
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
    
# Initialize the attack class, no training needed just results
attack = att.Attack(input_list, target_list, mask_list, 'adamax')

# Put together test images 
for image_path in input_test_path:
    input_test_list.append(att.detect_face(image_path)[0])

for image_path in target_test_path:
    target_test_list.append(att.detect_face(image_path)[0])

# Print results!
attack.results(input_test_list, target_test_list, '/experiment_2/')