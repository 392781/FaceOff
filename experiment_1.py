import adversarial_face_recognition as att
import glob
from tqdm import tqdm

"""
    Documentation for what is going on will come later

    For now this just takes 5 input IDs and 5 target IDs and then creates attacks for
    each one

    RESULTS: (all euclidean distances)
    inputs vs ground truths
    0.6919291615486145
    0.710687518119812
    0.896115243434906
    0.45093584060668945
    0.6631213426589966

    inputs vs target
    0.8771535158157349
    1.143986463546753
    1.1140056848526
    1.1364113092422485
    1.099700927734375

    inputs vs target 2
    0.8734384179115295
    1.3240668773651123
    0.9298726916313171
    1.2159967422485352
    0.8732156157493591

    target vs target 2
    0.5311635136604309
    0.6634145379066467
    0.7484209537506104
    0.45897066593170166
    0.7643409371376038

    adversarial vs ground truths
    0.8155664205551147
    1.2387598752975464
    1.0632591247558594
    1.137104868888855
    1.388970136642456

    adversarial vs target
    0.2924672067165375
    0.30148038268089294
    0.3449569642543793
    0.2755737900733948
    0.42731451988220215

    adversarial vs target 2
    0.6658462285995483
    0.6938043236732483
    0.8824158310890198
    0.5271591544151306
    0.9361984133720398
"""

# Gather all image locations
input_path = glob.glob('./faces/input/*.*')
target_path = glob.glob('./faces/target/*.*')
input_test_path = glob.glob('./faces/input_tests/*.*')
target_test_path = glob.glob('./faces/target_tests/*.*')

# Sort each image location by name so that each identity corresponds 
# for testing
input_path.sort()
target_path.sort()
input_test_path.sort()
target_test_path.sort()

# List initialization for the attack class inputs
input_list = []
target_list = []
input_test_list = []
target_test_list = []
mask_list = []

# Detect faces for all the image locations
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
    mask_list[i] = att.mask_offset(input_list[i], mask_list[i][0], mask_list[i][1])
    
# Create attack class and train using adamax
attack = att.Attack(input_list, target_list, mask_list, 'adamax')
attack.train(epochs = 45)

# Put together test images 
for image_path in input_test_path:
    input_test_list.append(att.detect_face(image_path)[0])

for image_path in target_test_path:
    target_test_list.append(att.detect_face(image_path)[0])

# Print results!
attack.results(input_test_list, target_test_list, '/experiment_1/')