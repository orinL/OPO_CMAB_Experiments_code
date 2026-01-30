"""
Usage: Set datasets to use and N_SHUFFLES below
"""

import argparse
import sys
import os
import subprocess

dataset_dir = 'datasets/' # directory of datasets to shuffle
output_dir = 'datasets_shuffled/' # directory to save shuffled datasets

datasets = ['ds_1006_2', 'ds_1012_2', 'ds_1015_2', 'ds_1062_2.0', 'ds_1073_2.0', 'ds_1084_3', 'ds_339_3', 'ds_346_2', 'ds_457_4', 'ds_476_2', 'ds_729_2', 'ds_785_2', 'ds_835_2', 'ds_848_2', 'ds_874_2', 'ds_905_2', 'ds_928_2', 'ds_964_2']

N_SHUFFLES = 10

if __name__ == '__main__':

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for ds in datasets:

        ds_path = os.path.join(dataset_dir, ds + ".vw.gz")
        if not os.path.exists(ds_path):
            print(ds_path, " does not exist.")
            continue


        for idx in range(N_SHUFFLES):

            shuf_name = ds + '_shuf{}'.format(idx)
            shuf_path = os.path.join(output_dir, shuf_name + ".vw.gz")

            if not os.path.exists(shuf_path):
                
                print("Writing to {}".format(shuf_path))
                
                # Use a simpler approach that works on the server
                try:
                    cmd = f'gunzip -c "{ds_path}" | shuf | gzip > "{shuf_path}"'
                    result = subprocess.run(cmd, shell=True, check=True)
                    print(f"Successfully created {shuf_path}")
                        
                except subprocess.CalledProcessError as e:
                    print(f"Error shuffling {ds_path}: {e}")
                    continue
