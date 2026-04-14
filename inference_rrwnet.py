import sys, json, os, time, random
import os.path as osp
import pathlib
import numpy as np
import torch
import tqdm
import pandas as pd
from PIL import Image
from AVS.modeling.RRWNet import RRWNet as RRWNetModel
from huggingface_hub import PyTorchModelHubMixin

from AVS.modeling.data_load import get_retinal_seg_test_loader
from AVS.modeling.inference import inference_rrwnet

class RRWNet(RRWNetModel, PyTorchModelHubMixin):
    def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=5):
        super().__init__(input_ch, output_ch, base_ch, iterations)

def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Inference for 2d Biomedical Image Segmentation')
    parser.add_argument('--csv_path_test', type=str, default='data/csvs/tr_scA_f1.csv', help='csv path test data')
    parser.add_argument('--problem_type', type=str, default='multi_label', help='problem type: multi_class, binary_class, multi_label')
    parser.add_argument('--load_path', type=str, default='', help='path to weight of pretrained model, if any')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers')
    parser.add_argument('--save_predictions', type=str2bool, nargs='?', const=True, default=True, help='whether to save predictions on test set')
    args = parser.parse_args()

    return args


def set_seeds(seed_value, use_cuda, use_mps, MAC:bool=False):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda and not MAC:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
    
    if MAC and use_mps:
        torch.mps.manual_seed(seed_value)


if __name__ == '__main__':

    args = get_args_parser()

    #1. Define device, seeds for reproducibility
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('* Using device {}'.format(device))

    seed_value = 0
    MAC = sys.platform == "darwin"
    set_seeds(seed_value, use_cuda, use_mps, MAC=MAC)

    #3. Gather parse arguments, create model, dataloaders, optimizer, loss
    problem_type = args.problem_type
    bs = args.batch_size
    csv_path_test, nw = args.csv_path_test, args.num_workers
    csv_df = pd.read_csv(csv_path_test)

    CLASS_DICT = {0: 'background', 1: 'artery', 2: 'vein',  3: 'crossings'}
    model = RRWNet(input_ch=3, output_ch=3, base_ch=64, iterations=5)
    state_dict = torch.load(args.load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    #List datasets
    dataset_list = csv_df['dataset'].unique()
    print('* Starting inference...')
    start_time = time.time()
    for dataset in dataset_list:

        dataset_path = pathlib.Path.cwd() / "datasets_organized" / dataset

        #dataloaders
        print('* Creating Dataloaders, batch size={}, workers={}'.format(bs, nw))
        test_loader = get_retinal_seg_test_loader(csv_path_test=csv_path_test, 
                                                batch_size=bs, 
                                                num_workers=nw, 
                                                problem_type=problem_type,
                                                dataset_name=dataset,
                                                tg_size=(1024,1024),
                                                ignore_od=False,
                                                resize_mode="rrwnet")  #RRWNet-specific code for image enhancement and padding

        metrics, subject_ids, predictions, img_paths = inference_rrwnet(model=model, loader=test_loader, CLASS_DICT=CLASS_DICT, device=device)

        #5. Save metrics
        rows = []
        for key in ["global", "zone_B", "junctions", "arcades_cldice"]:
            mean_vals, std_vals = metrics[key]
            # If these are arrays (per class), iterate over classes
            for i, (m, s) in enumerate(zip(mean_vals, std_vals)):
                rows.append({
                    "metric": key,
                    "class": i,
                    "mean": float(m) if not np.isnan(m) else None,
                    "std": float(s) if not np.isnan(s) else None
                })

        for junction, (mean_val, std_val) in metrics["junction_recalls"].items():
            rows.append({
                "metric": f"junction_recall_{junction}",
                "class": None,
                "mean": float(mean_val) if not np.isnan(mean_val) else None,
                "std": float(std_val) if not np.isnan(std_val) else None
            })

        faz_mean, faz_std = metrics["faz_tnr"]
        rows.append({
            "metric": "faz_tnr",
            "class": None,
            "mean": float(faz_mean) if not np.isnan(faz_mean) else None,
            "std": float(faz_std) if not np.isnan(faz_std) else None
        })

        # Convert to DataFrame and save CSV
        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(dataset_path / "rrwnet_retrained_metrics.csv", index=False)
        print("Metrics saved to rrwnet_retrained_metrics.csv")

        #6. Save predictions 
        if args.save_predictions:
            
            for subject_id, pred, img_path in zip(subject_ids, predictions, img_paths): #subject_id is dataset_sub_XXXX
                
                #First, work with numpy array
                #1. Ensure size is 1024x1024
                if len(pred.shape) == 3:
                    pred = pred[0,:,:]

                #2. Remove padding if any (RRWNet-specific code)
                original_image = Image.open(img_path)
                original_width, original_height = original_image.size
                if pred.ndim == 3:
                    H_pad, W_pad, _ = pred.shape
                else:
                    H_pad, W_pad = pred.shape

                pad_top = (H_pad - original_height) // 2
                pad_bottom = H_pad - original_height - pad_top
                pad_left = (W_pad - original_width) // 2
                pad_right = W_pad - original_width - pad_left

                if pred.ndim == 3:
                    pred = pred[pad_top:H_pad - pad_bottom, pad_left:W_pad - pad_right, :]
                else:
                    pred = pred[pad_top:H_pad - pad_bottom, pad_left:W_pad - pad_right]

                #3. Save 
                #Extract sub_id 
                if len(subject_id.split("_")) == 3: #dataset_sub_XXXX
                    sub_id = subject_id.split("_")[1] + "_" + subject_id.split("_")[2]

                elif len(subject_id.split("_")) == 4: #my_dataset_sub_XXXX
                    sub_id = subject_id.split("_")[2] + "_" + subject_id.split("_")[3]

                #We need to know the image number. Image paths are like .../sub_XXXX/sub_XXXX_img_cropped_resized.png if only one subject per image
                #or .../sub_XXXX/sub_XXXX_img_{number}_cropped_resized.png where number = 1, 2. If this number is there, we will use it in the save path to avoid overwriting.
                img_path = pathlib.Path(img_path)   
                img_name = img_path.stem  #sub_XXXX_img_cropped_resized or sub_XXXX_img_{number}_cropped_resized
                if "_1" in img_name:
                    img_number = "_1_"

                elif "_2" in img_name:
                    img_number = "_2_"
                
                else:
                    img_number = "_"

                dataset_seg_dir = pathlib.Path.cwd() / "datasets_organized" / dataset / sub_id / "rrwnet_retrained" 
                dataset_seg_dir.mkdir(parents=True, exist_ok=True)
                save_path_pred = dataset_seg_dir / f"{sub_id}{img_number}segmentation.png"
                
                #Modify values to save: 
                # 1) ignore labels as 0
                pred_to_save = np.copy(pred)
                pred_to_save[pred == 255] = 0  #255 is ignore index

                #2) Art is 1 (channel R), vein is 2 (channel B), crossings is 3 (channel G), background is 0; save as 3 channels
                pred_to_save_3ch = np.zeros((pred_to_save.shape[0], pred_to_save.shape[1], 3), dtype=np.uint8)
                for class_idx in range(1, len(CLASS_DICT)):
                    if class_idx == 1:  #artery -> R
                        pred_to_save_3ch[:,:,0][pred_to_save == class_idx] = 255
                    elif class_idx == 2:  #vein -> B
                        pred_to_save_3ch[:,:,2][pred_to_save == class_idx] = 255
                    elif class_idx == 3:  #crossings -> G
                        pred_to_save_3ch[:,:,1][pred_to_save == class_idx] = 255
                
                if problem_type == "binary_class":
                    pred_to_save = pred_to_save.astype(np.uint8) * 255  #binary mask as 0,255
                else:
                    pred_to_save = pred_to_save_3ch
                    
                pred_pil = Image.fromarray(pred_to_save.astype(np.uint8))
                pred_pil.save(save_path_pred)
        
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print('Done. Inference time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))