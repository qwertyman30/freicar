**Exercise 3 Notes:** 

Arguments for training

'--resume', default=None, help='path to latest checkpoint (default: none)'

'--start_epoch', default=0, help="Start at epoch X"

'--batch_size', default=8, help="Batch size for training"

'--num_epochs', default=50, help="Number of epochs for training"

'--eval_freq', default=10, help="Evaluation frequency"

'--print_freq', default=1000, help='print frequency (default: 1000)'

'--gamma_reg', default=10., help='Weighting for regression loss'

'--gamma_seg', default=4., help='Weighting for segmentation loss'


-ROS Node Exercise: </br>
  -run `publish_images.py`</br>
  -ros topic for regression image: `reg_image` </br>
  -ros topic for segmentation image: `seg_image`</br>
                   -ros topic for birdseye markers: `birdseye` </br>
</br>

-weights for trained fast scnn are in `saved_models` </br>
-all other modified code is in the provided python files (fast_scnn_model.py, train_segmentation.py)
