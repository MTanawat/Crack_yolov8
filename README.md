## Usage
install requirements.txt
## Arguments
--img_path: Path to the input image.
--model_path: Path to the YOLO model.
--patch_size: Number of patches in rows and columns. 
--output_image_path: Path to save the reconstructed image.

## Requirements
* Linux (Ubuntu)
* Python = 3.9
* Pytorch = 1.13.1
* NVIDIA GPU + CUDA CuDNN
  
Example Command
 ```
python test.py --img_path "path/to/your/image.jpg" --model_path "path/to/your/model.pt" --patch_size 7 --output_image_path "path/to/save/reconstructed_image.jpg"
 ```
