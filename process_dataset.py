import os
import cv2
from Preprocessing import preprocess_image
from shutil import copyfile

def process_dataset(input_root, output_root):
    # Ensure the output directory exists, create if not
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Iterate through classes in the original dataset
    for class_name in os.listdir(os.path.join(input_root, 'train')):
        class_folder = os.path.join(input_root, 'train', class_name)
        output_class_folder = os.path.join(output_root, 'train', class_name)

        # Ensure the output class directory exists, create if not
        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        try:
            # Iterate through images in the class folder, excluding hidden files
            for file_name in [f for f in os.listdir(class_folder) if not f.startswith('.')]:
                if file_name.endswith('.jpg'):
                    # Process the original image and get annotations
                    original_image_path = os.path.join(class_folder, file_name)
                    original_preprocessed_image, annotations = preprocess_image(original_image_path)

                    # Save the original processed image
                    output_original_image_path = os.path.join(output_class_folder, file_name)
                    cv2.imwrite(output_original_image_path, original_preprocessed_image)

                    # Save the original annotations
                    output_original_annotation_path = os.path.join(output_class_folder, f'{os.path.splitext(file_name)[0]}.txt')
                    with open(output_original_annotation_path, 'w') as f:
                        for annotation in annotations:
                            f.write(' '.join(map(str, annotation)) + '\n')

                    # Process and save augmented image
                    augmented_preprocessed_image, _ = preprocess_image(original_image_path)

                    # Save the augmented image
                    output_augmented_image_path = os.path.join(output_class_folder, f'aug_{file_name}')
                    cv2.imwrite(output_augmented_image_path, augmented_preprocessed_image)

        except Exception as e:
            print(f"Error processing images in class folder: {class_folder}")
            print(f"Error details: {e}")

    print("Dataset processing completed.")

input_dataset_root = '/Users/albagir/Desktop/FinalProject/Traffic_signs-2'
output_dataset_root = '/Users/albagir/Desktop/FinalProject/Traffic_signs-2'
process_dataset(input_dataset_root, output_dataset_root)
