import adversarial_face_recognition as att
import glob
from tqdm import tqdm

"""
    Documentation for what is going on will come later

    Another test of transferability except this tests targeting a different target
    image... Didn't use this in results since this experiment wasn't well thought 
    out

    RESULTS: (all euclidean distances)
    inputs vs ground truths
    0.6919291615486145
    0.710687518119812
    0.896115243434906
    0.45093584060668945
    0.6631213426589966

    inputs vs target
    0.8734384179115295
    1.3240668773651123
    0.9298726916313171
    1.2159967422485352
    0.8732156157493591

    inputs vs target 2
    0.8771535158157349
    1.143986463546753
    1.1140056848526
    1.1364113092422485
    1.099700927734375

    target vs target 2
    0.5311635136604309
    0.6634145379066467
    0.7484209537506104
    0.45897066593170166
    0.7643409371376038

    adversarial vs ground truths
    0.8526229858398438
    0.9455323815345764
    0.8666716814041138
    0.5481076240539551
    0.8349324464797974

    adversarial vs target
    1.0066981315612793
    1.281568169593811
    1.0845149755477905
    1.1506640911102295
    0.9421268701553345

    adversarial vs target 2
    0.9942540526390076
    1.0539507865905762
    1.1439930200576782
    1.0762476921081543
    1.143889307975769
"""

input_path = glob.glob('./faces/input/*.*')
target_path = glob.glob('./faces/target_tests/*.*')
mask_path = glob.glob('./results/experiment_1/delta/*.*')
input_mask_path = glob.glob('./results/experiment_1/input/*.*')
input_test_path = glob.glob('./faces/input_tests/*.*')
target_test_path = glob.glob('./faces/target/*.*')

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

attack.results(input_test_list, target_test_list, '/experiment_3/')