import logging
import json
import os

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import Dataset
def create_sam_predictor():
    model_config = "vit_b"
    model_weights_path = (
        "/content/sam_checkpoints/sam_vit_b.pth"
    )

    sam = sam_model_registry[model_config](
        checkpoint=model_weights_path
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    sam.to(device)
    sam.eval()

    return SamPredictor(sam)
log =  logging.getLogger(__name__) #tạo logger có tên là helper

def run(app_cls, *argv):#run(LunaTrainingApp, '--epochs=1')
    #app_cls là class app tôi muốn chạy, *argv là các tham số truyền thêm
    argv = list(argv)#biến tuple thành list argv =['--epochs=1']
    argv.insert(0, '--num-workers=1') #argv = ['--num-workers=4', '--epochs=1']

    log.info(f"Running: {app_cls.__name__}({argv!r}).main()")
    #Running: LunaTrainingApp(['--num-workers=4', '--epochs=1']).main()
    
    app_cls(argv).main()
    log.info(f"Finished: {app_cls.__name__}({argv!r}).main()")

def normalize_tensor(tensor):
    """
    Chuẩn hóa tensor về khoảng [0, 255] và chuyển thành uint8.
    """
    min_val = tensor.min()
    max_val = tensor.max()

    value_range = max_val - min_val

    if value_range == 0:
        return torch.zeros_like(tensor, dtype=torch.uint8)
    normalized_tensor = (tensor - min_val) / value_range
    return (
        normalized_tensor
        .mul(255)
        .clamp(0, 255)
        .to(torch.uint8)
    )#normalize về 0->255 kiểu torch unit8
    #vì hu có giá trị -1000 -> 1000, chuyển vậy để lưu bằng PIL.Image
fine_tuning_dir = "/content/gdrive/MyDrive/luna_dataset/fine_tuning_dataset"
ct_folder = f"{fine_tuning_dir}/ct"
mask_folder = f"{fine_tuning_dir}/mask"
metadata_folder = fine_tuning_dir
def generate_ct_images_and_masks(
    original_ct_data, max_dataset_size = 500,
    #original_ct_data là tuple có dạng (ct_slice, label, series_uid, center_irc)
    recompute= False#recompute dùng để quyết định có tạo lại toàn bộ dataset hay không.
):
    """
    Tạo:
        fine_tuning_dir/
        ├── ct/
        ├── mask/
        └── metadata.jsonl

    original_ct_data phải trả về mỗi sample:
        ct_slice, label, series_uid, center_irc

    Hàm sử dụng biến sam đã được load ở bên ngoài.
    """
    os.makedirs(ct_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)
    metadata_filepath = os.path.join(
        metadata_folder,"metadata.jsonl"
        #"data-unversioned/part2/fine-tuning/dataset/metadata.jsonl"
    )
    existing_metadata = set()
    if recompute:
        for folder in (ct_folder, mask_folder):
            for filename in os.listdir(folder):
                if filename.endswith(".png"):
                    os.remove(
                        os.path.join(folder, filename)
                    )
    if not recompute and os.path.exists(metadata_filepath):# nếu ko tạo lại dataset và đã có metadata file
        with open(metadata_filepath,"r", encoding = "utf-8") as metadata_file:
            for line in metadata_file:
                line = line.strip() # danh sách mỗi dòng trong file

                if not line: # nếu dòng đó ko có chữ nào
                    continue # chạy tiếp
                row= json.loads(line)
                #{"index": 0, "series_uid": "1.3.6.1.4.1...", "center_irc": [91, 360, 341], "ct_file_name": "ct/0.png", "mask_file_name": "mask/0.png"}
                existing_metadata.add(int(row["index"]))
                #existing_metadata = {0, 1, 2, ...}
    file_mode = "w" if recompute else "a" #nếu tạo lại dataset thì ghi đè, ko thì append
    

    with open(
        metadata_filepath,
        file_mode,
        encoding="utf-8",
    ) as metadata_file:
        predictor = create_sam_predictor()
        for ct_data_index, sample_tup in enumerate(original_ct_data):# duyệt qua từng data
            if len(existing_metadata) >= max_dataset_size:
                print(
                    f"finished: da tao xong"
                    f"{len(existing_metadata)} samples"
                )
                break
            
            ct_slice, _, series_uid, center_irc = sample_tup
            #tensor, tensor, str, tensor
            #_ là label ko cần xài

            # Khi chạy trực tiếp từ Dataset, các tensor thường ở CPU.
            # Dòng này vẫn giúp code an toàn nếu tensor đang ở GPU.
            if torch.is_tensor(ct_slice):
                ct_slice = ct_slice.detach().cpu()#tách khỏi đồ thị gradient
            if torch.is_tensor(center_irc):
                center_irc_list = (
                    center_irc
                    .detach()
                    .cpu()
                    .tolist()
                )
            else:
                center_irc_list = list(center_irc)
            center_irc_list = [int(value) for value in center_irc_list]
            #chuyển thành python int mới json dump được
            #center_irc_list là python list chứa tâm irc
            # Đã tạo sample này rồi thì bỏ qua.
            if not recompute and ct_data_index in existing_metadata:
                continue
            ## CT là ảnh grayscale nhưng SAM nhận ảnh RGB.
            #1 lát ct hỉ chứa giá trị Hounsfield Unit (HU) tại mỗi pixel. phải chuyển thành RGB
            if ct_slice.ndim == 2: #dim=2 (H,W) là grayscale
                ct_slice_rgb =(
                    ct_slice
                    .unsqueeze(0) # thêm chiều đầu tiên (1,H,W)
                    .repeat(3,1,1)
                    #repeat(3, 1, 1) nghĩa là lặp tensor theo từng chiều:
                    #chiều channel lặp 3 lần, 2 chiều còn lại ko lặp
                )
            elif ct_slice.ndim == 3 and ct_slice.shape[0] == 1:
                ct_slice_rgb = ct_slice.repeat(3, 1, 1)
            elif ct_slice.ndim == 3 and ct_slice.shape[0] == 3:
                ct_slice_rgb = ct_slice
            else:
                raise ValueError(
                    "ct_slice phải có shape [H, W], "
                    "[1, H, W] hoặc [3, H, W]. "
                    f"Shape hiện tại: {tuple(ct_slice.shape)}"
                )
            scaled_ct_slice = normalize_tensor(ct_slice_rgb)
            #trả về tensor đã clamp về 0->255 và unit8

            ## Chuyển [C, H, W] thành [H, W, C] cho PIL và SAM.
            ct_image_array= (
                scaled_ct_slice.
                permute(1,2,0)
                .contiguous()#permute đổi cách nhìn dữ liệu nhưng khi đọc
                #thì sẽ đọc dữ liệu ko contiguous, xài contiguous() để copy dữ
                #liệu sang vùng mới sẽ liên tục
                .numpy()
            )
            ct_image = Image.fromarray(# chuyển thành ảnh PIL
                ct_image_array,
                mode="RGB",
            )#sau khi chuyển có thể .save hoặc .show
            filename = f"{ct_data_index}.png" # ví dụ 0.png

            ct_filepath = os.path.join(
                ct_folder, 
                filename,
            )#"data-unversioned/part2/fine-tuning/dataset/ct/0.png"
            mask_filepath = os.path.join(
                mask_folder,
                filename,
            )#"data-unversioned/part2/fine-tuning/dataset/mask/0.png"

            # center_irc có thứ tự:
            # [index, row, column]
            # SAM nhận tọa độ:
            # [x, y] = [column, row]
            #center_irc_list là mảng numpy chứa tâm irc
            x = float(center_irc_list[2])
            y = float(center_irc_list[1])
            input_points = np.array(
                [[x,y]],
                dtype = np.float32
            )#mảng 2 chiều, type là float32
            predictor.set_image(ct_image_array)# đưa vào tính trước image embedding.

            masks, iou_predictions, logits = predictor.predict(
                point_coords = input_points,
                point_labels = np.array([1]),
                multimask_output = False,
            )# masks shape là (1, 720, 1280), logits shape là (1, 256, 256)
            #iou_predictions.shape là (1,)
            mask_image_array = masks[0]
            mask_area = int(mask_image_array.sum())
            if not (
                0 < mask_area < 1100#lọc bỏ mask quá lớn
                and float(iou_predictions[0]) > 0.88 # và iou quá thấp
            ):
                continue
             # Với NumPy bool, PIL tạo mask nhị phân.

            mask_image_array = (
                masks[0].astype(np.uint8) * 255
            )#mode L tương thích các thư viện tốt hơn
            mask_image = Image.fromarray(mask_image_array,mode ='L')

            ct_image.save(ct_filepath) #ảnh PIL RGB
            ##"data-unversioned/part2/fine-tuning/dataset/ct/0.png"
            mask_image.save(mask_filepath) # ảnh PIL HW
            #"data-unversioned/part2/fine-tuning/dataset/mask/0.png"

            relative_ct_path = os.path.relpath(
                ct_filepath,#fine_tuning_dir = "data-unversioned/part2/fine-tuning/dataset"
                start=fine_tuning_dir,#data-unversioned/part2/fine-tuning/dataset/ct/15.png"
            ).replace("\\", "/")
            #output ra ct\\15.png replace thành ct/15.png
            relative_mask_path = os.path.relpath(#mask/0.png
                mask_filepath,
                start=fine_tuning_dir,
            ).replace("\\", "/")

            metadata_item = {
                "index": int(ct_data_index), 
                "series_uid": str(series_uid),
                "center_irc": [int(x) for x in center_irc_list],
                "ct_file_name": str(relative_ct_path),
                "mask_file_name": str(relative_mask_path),
            }
            metadata_file.write(
                json.dumps(metadata_item) +"\n"
            )
            metadata_file.flush()#dùng để ép dữ liệu đang nằm trong bộ đệm ghi xuống file ngay lập tức.
            existing_metadata.add(ct_data_index)


       

