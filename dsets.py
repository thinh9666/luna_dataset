import functools
import collections
import copy
import csv
import glob
import torchio as tio

import numpy as np
import torch
import random
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset
mhd_data_folder = "/content/luna_data/subset0"

CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple",
    "isNodule_bool, diameter_mm, series_uid, center_xyz"
)

IrcTuple = collections.namedtuple("IrcTuple", ["index", "row", "col"])
XyzTuple = collections.namedtuple("XyzTuple", ["x", "y", "z"])


@functools.lru_cache(1)
def getCandidateInfoToList(require_on_disk=True):# train gọi 1 lần val gọi 1 lần nên lru cache 1  là đủ
    data_directory = Path(f"{mhd_data_folder}")
    mhd_files = list(Path(mhd_data_folder).glob("*.mhd"))
    #trả về danh sách các đường dẫn có chứa đuôi .mhd
    if not mhd_files:
        print(f"no mhd files found under {data_directory} directory")
    present_on_disk_set = {path.stem for path in mhd_files}
    #{} trả về set các tên file có đuôi mhd
    #hàm stem là hàm của Path nhưng bỏ đi đuôi mở rộng

    diameter_dict = {}
    with open('dlwpt-code-2e/data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:#bỏ dòng header đầu
        #csv.reader(f) trả về iterator, nên phải bọc list
        #row là list các giá trị
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (annotation_center_xyz, annotationDiameter_mm)
            )
            #setdefault nếu chưa có key thì tạo key với list rỗng
        '''
        { thuộc về annotations
            "abc123":[
                ((12.5,34.2,56.7), 8.1),
                ((22.1,50.3,61.2), 5.4)
            ]
        }
        '''
        candidate_info_list = []
        with open('dlwpt-code-2e/data/part2/luna/candidates.csv', 'r') as f:
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]
                if series_uid not in present_on_disk_set and require_on_disk:
                    continue
                isNodule_bool = bool(int(row[4]))#string phải cast sang int rồi mới bool
                candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
                candidateDiameter_mm = 0.0
                for annotation_tup in diameter_dict.get(series_uid, []): #không có thì trả về []
                    #duyệt qua các center và diametter
                    annotation_Center_xyz, annotationDiameter_mm = annotation_tup
                    for i in range(3):# duyệt qua tọa độ 3 chiều x,y,z
                        delta_mm = abs(candidateCenter_xyz[i] - annotation_Center_xyz[i])
                        if delta_mm > annotationDiameter_mm/4:
                            break
                    else: # else của vòng for, chỉ chạy khi vòng for không bị break
                        candidateDiameter_mm = annotationDiameter_mm
                        #nếu độ lệch 3 trục đáp ứng tiêu chuẩn, thì coi andidate này và annotation này là cùng một nodule
                        #lấy candidateDiameter_mm gắn cho annotationDiameter_mm
                        break#break để qua
                candidate_info_list.append(CandidateInfoTuple(isNodule_bool,
                                                              candidateDiameter_mm,
                                                              series_uid,
                                                              candidateCenter_xyz))
        candidate_info_list.sort(reverse=True)
        #tuple sort theo tứ tự các phần tử
        #ưu tiên isNodule_bool=True, candidateDiameter_mm lớn
        #
        return candidate_info_list# list các CandidateInfoTuple


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    #coord_irc là tọa độ voxel trong array: (index, row, column)
    #origin_xyz Là gốc tọa độ thật của CT scan trong patient coordinate system: (X, Y, Z)(C,R,I) tính bằng mm
    #vxSize_xyz Là kích thước voxel theo từng trục thật:voxel size theo X, Y, Z (C,R,I)
    #direction_a Là ma trận hướng 3×3,cho biết trục của CT array có cùng hướng, bị lật, hoặc bị xoay so với patient coordinate system hay không.
    #[
    #[1, 0, 0], nếu -1 thì đổi hướng
    #[0, 1, 0],
    #[0, 0, 1],
    #]
    cri_a = np.array(coord_irc)[::-1] #(i,r,c) sang (c,r,i)
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    #origin_a là gốc tọa độ của CT trong hệ tọa độ bệnh nhân
    #array([160, 260, 340])
    return XyzTuple(*coords_xyz)
def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
  #coord_xyz tọa độ thật trong patient coordinate
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    #coord_a - origin_a: Lấy tọa độ XYZ thật trừ gốc CT array. Kết quả là: điểm đó cách gốc array bao nhiêu mm.
    cri_a = np.round(cri_a)#làm tròn đến số nguyên gần nhất
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))#đảo ngược lại thành I,R,C

class Ct:
  def __init__(self,series_uid):
      import SimpleITK as sitk
      mhd_path = glob.glob(f'{mhd_data_folder}/{series_uid}.mhd')[0]
      #vì glob.glob luôn trả về một list, dù chỉ 1 file
      ct_mhd = sitk.ReadImage(mhd_path)
      #trả về SimpleITK Image object metadata, chứa ảnh CT + thông tin tọa độ.
      ct_a = np.array(sitk.GetArrayFromImage(ct_mhd),dtype=np.float32)
      #mảng voxel chứa giá trị HU
      #chuyển nó thành NumPy array để mình xử lý bằng Python/PyTorch.
      # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
      ct_a.clip(-1000, 1000, ct_a)

      self.series_uid = series_uid
      self.hu_a = ct_a
      self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())#gốc tọa độ  của CT scan trong patient coordinate system
      self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())#kích thước của mỗi voxel, đơn vị mm/voxel
      self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)#ma trận hướng của CT scan. Nó cho biết các trục của mảng CT có cùng hướng với trục XYZ hay bị đảo/xoay
  def getRawCandidate(self, center_xyz, width_irc):
    #width_irc chứa thông tin số lượng voxel mỗi trục
    # ví dụ width_irc = (32, 48, 48)
      center_irc = xyz2irc(
          center_xyz,
          self.origin_xyz,
          self.vxSize_xyz,
          self.direction_a,
      )# trả về IrcTuple, chứa tọa độ I,R,C
      #ví dụ ra center_irc = (91, 360, 341)
      slice_list = []
      for axis, center_val in enumerate(center_irc):
          start_ndx = int(round(center_val - width_irc[axis]/2))#75,336,317
          end_ndx = int(start_ndx + width_irc[axis])#107,384,365
          
          if start_ndx < 0: # bé hơn 0 thì cho thành 0
              start_ndx = 0
              end_ndx = start_ndx + width_irc[axis]
          if end_ndx > self.hu_a.shape[axis]:
              end_ndx = self.hu_a.shape[axis]
              start_ndx = end_ndx - width_irc[axis]
          slice_list.append(slice(start_ndx, end_ndx))
            #slice_list = [
            #slice(75, 107),
            #slice(336, 384),
            #slice(317, 365)
            #]
      ct_chunk = self.hu_a[tuple(slice_list)]#tuple các slice
      return ct_chunk, center_irc # trả về chunk đã crop và trung tâm chunk
  
@functools.lru_cache(10, typed=True)
def getCt(series_uid):
  return Ct(series_uid)
@functools.lru_cache(maxsize=1)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid) # lấy ct có series_uid đó
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc #trả về ct chunk và tâm đã chuyển thành irc

def getCtAugmentedCandidate(augmentation, series_uid, center_xyz, width_irc,use_cache=True):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
        #ct_chunk shape là [depth, height, width]

    ct_t =torch.tensor(ct_chunk).unsqueeze(0).to(torch.float32)
    #thêm chiều   channel [1,depth,height,width]
    subject = tio.Subject(
        ct = tio.ScalarImage(tensor = ct_t)
    )
    augmented_subject = augmentation(subject)
    # augmented_subject.ct.data vẫn có shape [1, D, H, W]
    augmented_chunk_t = augmented_subject.ct.data
    augmented_chunk_t = augmented_chunk_t.clamp(-1000,1000)#
    # giống clip ở numpy
    return augmented_chunk_t, center_irc
class LunaDataset(Dataset):
    def __init__(self,
        val_stride=0,
        isValSet_bool=None,
        series_uid=None,
        ratio_int = 0, #chia đều negative và positive
        augmentation_bool=False,
        ):
        self.augmentation = None
        if augmentation_bool:
            self.augmentation = tio.Compose([
                tio.RandomFlip(
                    axes=(0, 1, 2),#0=depth,height,width, ko có chiều ko gian
                    flip_probability=0.5# mỗi trục lật  xác suất 0.5
                    #p=0.5 thì toàn bộ lật xác suất 0.5
                ),
                tio.RandomAffine(
                    scales=(0.9, 1.1),#phóng to thu nhỏ ngẫu nhiên
                    degrees=(10,0,0), # xoay ngẫu nhiên 10 độ, giữ nguyên depth, chỉ xoay width,height
                    translation=3,
                    p=0.75 # dịch chuyển ảnh tối đa 3 voxel
                ),
                tio.RandomNoise(
                    std=(0,25), # noise từ 0 ->25
                    p=0.25 #25% sample được thêm noise
                ),
            ])
        else:
            self.augmentation = None
        self.candidateInfo_list = copy.copy(getCandidateInfoToList())
        #phải copy để ko trỏ thẳng list gốc trong cache

        if isValSet_bool:
            assert val_stride > 0 ,val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0: # trường hợp ko phải val thì xóa các dữ liệu val
            del self.candidateInfo_list[::val_stride] #xóa xong các phần tử các sẽ dồn lên
            assert self.candidateInfo_list
        #chia train-validation trước rồi mới lọc uid
        
        if series_uid:# chỉ giữ các ct có uid nằm trong series_uid list
            if isinstance(series_uid, str):
                series_uid_set = {series_uid}
            else:
                series_uid_set = set(series_uid)

            self.candidateInfo_list = [
                x for x in self.candidateInfo_list
                if x.series_uid in series_uid_set
            ]
        

 
        self.ratio_int = ratio_int
        self.negative_list = [
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool # nt là namedtuple
        ] # danh sách các candidateInfo_list mà là negative
        self.positive_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ] # danh sách các candidateInfo_list mà là positive
    
    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)  
    def __len__(self):
        if self.ratio_int:
            return 5000
        else:
            return len(self.candidateInfo_list)
    #số sample 1 epoch dựa vào số __len__ trả về

    def __getitem__(self, index):
        '''
        lấy 1 candidate
        → xem candidate đó thuộc CT scan nào
        → load CT scan đó nếu chưa có trong cache
        → crop 1 vùng 32x48x48
        → trả sample
        '''
        if self.ratio_int: # ví dụ ratio_int = 2 là + - - + - - + #4
            pos_index = index // (self.ratio_int + 1) # 4// (2+1) = 1
            if index % (self.ratio_int +1) > 0: #ví dụ ratio_int =2 mà index = 4  nghĩa là negative
                neg_index = index - 1 - pos_index # 4 -1 - 1 = 2
                neg_index = neg_index % len(self.negative_list) # vì negative_list < len(dataset) nên làm vậy để ko bị tràn
                candidateInfo_tup = self.negative_list[neg_index]
            else: # positive nếu  index % (self.ratio_int +1) = 0
                pos_index = pos_index % len(self.positive_list)
                candidateInfo_tup = self.positive_list [pos_index]
        else:
            candidateInfo_tup = self.candidateInfo_list[index]
        width_irc = (32,48,48)
        if self.augmentation is not None:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )#đã thêm channel dimesnsion từ trước nên ko cần unsqueeze
        else:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )

            candidate_t = torch.from_numpy(
                candidate_a
            ).to(torch.float32).unsqueeze(0)
        #trả về ct_chunk và tâm irc
        # candidate_a shape: (32, 48, 48)
        # candidate_t shape sau unsqueeze: (1, 32, 48, 48)
        # số 1 là channel dimension

        # Label dạng 2 class:
        # [1, 0] = không phải nodule
        # [0, 1] = là nodule
        pos= torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool,
        ],dtype=torch.long)
        return (
            candidate_t, # chunk đã được crop
            pos, #label [0,1] hoặc [1,0]
            candidateInfo_tup.series_uid,#uid
            torch.tensor(center_irc),#tâm đã chuyển sang irc
        )
    

