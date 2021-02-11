from adversarial_face_recognition import *

# Load the data.  This will detect and resize the faces
input_list = load_data('./faces/input/*.*')
target_list = load_data('./faces/target/*.*')

# A list for calculated face masks
mask_list = []

# Create the masks for the inputs
for face, _ in tqdm(input_list):
    mask_list.append(create_mask(face))

# If training on differing masks and inputs (i.e. mask ID doesn't match input ID)
# offset the mask
for i in tqdm(range(len(mask_list))):
    mask_list[i] = mask_offset(input_list[i], mask_list[i][0], mask_list[i][1])

# Create attack class and train using adamax
attacker = Attack(input_list, target_list, mask_list, 'adamax')
attacker.train(epochs=45)

# For result testing load the image testing set
input_test_list = load_data('./faces/input_tests/*.*')
target_test_list = load_data('./faces/target_tests/*.*')

# Print your results.  Make sure you've created the save directory beforehand
attacker.results(input_test_list, target_test_list, '/results/')