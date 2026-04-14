python inference_rrwnet.py \
    --csv_path_test data_splits/artery_vein_segmentation/test_av_segmentation.csv \
     --load_path models/rrwnet_HRF_0.pth \
     --problem_type multi_label \
     --batch_size 4
