import os
import csv

data_dir = '/datapath'

csv_dir = data_dir
os.makedirs(csv_dir, exist_ok=True)

splits = ['train', 'val', 'test']

for split in splits:
    split_dir = os.path.join(data_dir, split)
    csv_file = os.path.join(csv_dir, f'{split}.csv')
    
    print(f'Creating or overwriting CSV file for {split} split: {csv_file}')
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])
        
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(split, class_name, img_name)
                        writer.writerow([img_path, class_name])
                        print(f'Added {img_path} to {csv_file}')

for split in splits:
    csv_file = os.path.join(csv_dir, f'{split}.csv')
    if os.path.exists(csv_file):
        print(f'{csv_file} exists.')
    else:
        print(f'{csv_file} doesnot exist.')