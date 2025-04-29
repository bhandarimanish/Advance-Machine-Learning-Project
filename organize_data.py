import os
import shutil

# Where your files are currently
source_folder = '/Users/manishbhandari/Downloads/thinning_data_1/data/thinning' 
# Where you want the organized copies to go
destination_base = '/Users/manishbhandari/Documents/Advanced ML/aml_project/data'  
# Define target folders
input_folder = os.path.join(destination_base, 'input')
geojson_folder = os.path.join(destination_base, 'geojson')
output_folder = os.path.join(destination_base, 'target')

# Create folders if they don't exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(geojson_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Process files
for file_name in os.listdir(source_folder):
    source_file = os.path.join(source_folder, file_name)

    # Copy input images
    if file_name.startswith('image_') and file_name.endswith('.png'):
        shutil.copy2(source_file, os.path.join(input_folder, file_name))
    
    # Copy geojson files
    elif file_name.startswith('target_') and file_name.endswith('.geojson'):
        shutil.copy2(source_file, os.path.join(geojson_folder, file_name))
    
    # Copy output target images
    elif file_name.startswith('target_') and file_name.endswith('.png'):
        shutil.copy2(source_file, os.path.join(output_folder, file_name))

print("All files copied successfully!")
