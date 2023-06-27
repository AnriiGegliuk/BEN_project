import os
import logging
import warnings
import tensorflow as tf

from utils.inference import inference_pipeline
from utils.check_result import make_result_to_logs
from utils.check_html import make_logs_to_html
from utils.new_mask import update_header_and_save

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # tf log errors only
logging.getLogger('tensorflow').setLevel(logging.ERROR)
print(tf.__version__)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='input', required=True, type=str, help="Input folder")
    parser.add_argument("-o", dest='output', required=True, type=str, help="Output folder")
    parser.add_argument("-new_mask_dir", dest='new_mask_dir', required=True, type=str, help="New Mask folder")
    parser.add_argument("-weight", dest='weight', help="model weight path",
                        default=r'weight/unet_fp32_all_BN_NoCenterScale_polyic_epoch15_bottle256_04012056/')
    parser.add_argument("-check", dest='check_orientation',
                        help="Check input orientation. None for skipping. 'RIA' for rodents and 'RPI' for NHPs")
    parser.add_argument("-mkdir", dest='is_mkdir', default=True, help="If the output folder doesn't exist, creat it")

    parser.set_defaults(BN_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    args = parser.parse_args()

    ''' Run inference '''
    inference_pipeline(args.input, args.output, weight=args.weight, BN_list=args.BN_list,
                       check_orientation=args.check_orientation, is_mkdir=args.is_mkdir)

    ''' Generate visual report '''
    logs_folder = make_result_to_logs(input_folder=args.input, predict_folder=args.output, orientation=args.check_orientation)
    make_logs_to_html(log_folder=logs_folder)  # HTML logs will be saved in this folder

    ''' Generate new masks '''
    raw_files = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)) and f.endswith('.nii')]

    if not os.path.exists(args.new_mask_dir):
        os.makedirs(args.new_mask_dir)

    for raw_file in raw_files:
        raw_filepath = os.path.join(args.input, raw_file)
        mask_filepath = os.path.join(args.output, raw_file) 
        new_mask_filepath = os.path.join(args.new_mask_dir, os.path.splitext(raw_file)[0] + '_mask.nii')
        update_header_and_save(raw_filepath, mask_filepath, new_mask_filepath)

    print('\n**********\t', f'Completed. New masks generated & saved in {args.new_mask_dir}', '\t**********\n')


""" Comman do run the script is: python BEN_infer_new.py -i (your input folder for raw images) -o (your output folder for generated masks) -new_mask_dir (your new folder of transformed maks) -weight weight/exvivo_scan_06251403/.hdf5 -check RIA"""