
import sys
import argparse
import datetime
import torch
import logging
from torch.optim import SGD
from model import LunaModel
from dsets import LunaDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
import os
METRICS_LABEL_NDX = 0 # label thật các sample
METRICS_PRED_NDX = 1 # xác suất label dự đoán các sample 
METRICS_LOSS_NDX = 2 # loss của các sample
METRICS_SIZE = 3
log = logging.getLogger(__name__)
log_train = logging.getLogger("training")
log_val = logging.getLogger("validation")
def settingLogging(log_dir="/content/luna_dataset/logs"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True,exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s pid:%(process)d "
        "%(name)s:%(lineno)d:%(funcName)s %(message)s"
    )#2026-06-17 08:30:12,123 INFO pid:3821 training:145:main E1 trn 0.4321 loss, 98.5% correct
    root_logger = logging.getLogger()# lấy root logger, logger cha cao nhất
    root_logger.setLevel(logging.INFO)# chỉ nhận log từ info trở lên, bỏ debug
    root_logger.handlers.clear() #xóa handler cũ để tránh log bị lặp
    #Nếu không xóa handler cũ thì mỗi lần bạn gọi lại setupLogging() nó sẽ add thêm handler mới chồng lên handler cũ.
 
    console_handler = logging.StreamHandler()#dùng để in log ra màn hình/console, thay vì ghi vào file.
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)#  root logger sẽ in ra console

    train_handler = logging.FileHandler(log_dir/"training.log",mode="a")
    train_handler.setLevel(logging.INFO)
    train_handler.setFormatter(formatter)

    val_handler = logging.FileHandler(log_dir/"validation.log",mode="a")
    val_handler.setLevel(logging.INFO)
    val_handler.setFormatter(formatter)

    train_log = logging.getLogger("training")
    val_log = logging.getLogger("validation")
    train_log.setLevel(logging.INFO)# set level mức đầu
    val_log.setLevel(logging.INFO)

    train_log.handlers.clear()#xóa hết handler đang gắn trực tiếp vào train_log và val_log.
    val_log.handlers.clear()

    train_log.addHandler(train_handler)
    val_log.addHandler(val_handler)
    train_log.propagate = False # ko propagate tới logger root
    val_log.propagate = False

class LunaTrainingApp: 
    def __init__(self, sys_argv = None):
        #python -m p2ch13.training --num-workers=4 --epochs=1
        if sys_argv is None:
        #nếu người dùng ko đưa vào argv thì lấy argv từ command line
            sys_argv = sys.argv[1:] #[--num-workers=4 , --epochs=1]
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help = 'Number of worker processes for background data loading',
                            default =8,
                            #có -- là optional argument, ko có thì là positional argument
                            type=int)#khi thấy arg num worker thì thấy giá trị int ngay sao nó
        parser.add_argument('--batch-size', help='batch size used for training',default=32,type=int)
        parser.add_argument('--epochs',help='Number of epochs to train for',default=1,type=int,)
        parser.add_argument('--balanced',
                            help = "Balance the training data to half positive",
                            action="store_true",default=False)
                    #store_true nghĩa là nếu xuất hiện balanced thì giá trị của balanced sẽ là true, ko cần viết --balanced true
                    #default=false là nếu ko nhập balanced sẽ là false
        #line63
        self.cli_args = parser.parse_args(sys_argv) # nếu tham số là --help thì in ra rồi dừng luôn
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model= self.initModel()
        self.optimizer = self.initOptimizer()
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None
        self.checkpoint_path = "/content/gdrive/MyDrive/luna_checkpoints/luna_latest.pt"
        self.start_epoch = 1
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.totalTrainingSamples_count = checkpoint["totalTrainingSamples_count"]
            self.start_epoch = checkpoint["epoch"] + 1

            log.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def initModel(self):
        model =LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA, {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                #thì DataParallel sẽ chia batch ra cho nhiều GPU xử lý cùng lúc.
            model = model.to(self.device)
        return model
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr = 0.001, momentum=0.99)
    def chunks(self, lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]
    def initTrainDl(self,series_uid_batch = None):
        train_ds = LunaDataset(
            val_stride = 10,
            isValSet_bool = False,
            ratio_int = int(self.cli_args.balanced),#ratio_int mặc định là 1
            augmentation_bool=True,
            series_uid=series_uid_batch,
        )
        if self.cli_args.balanced and len(train_ds.positive_list) == 0:
            return None
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            #làm vậy mỗi gpu nhận 1 batch
        
        train_dl = DataLoader(
            train_ds,
            batch_size = batch_size,
            num_workers = self.cli_args.num_workers,
            pin_memory = self.use_cuda,#
        )
        return train_dl
    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool= True

        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        val_dl = DataLoader(
            val_ds,
            batch_size = batch_size,
            num_workers = self.cli_args.num_workers,
            pin_memory = self.use_cuda,
            #dành vài pages trong ram, khóa lại, lúc đó batch A ở địa chỉ ox.... sẽ ko bị thay đổi
            #=> GPU DMA có thể đọc trực tiếp
            # vì địa chỉ ko đổi, CPU ko cần phải kiểm tra địa chỉ, điều phối, đôi khi coppy qua buffer trung gian tốn thời gian
        )
        return val_dl
    def computeBatchLoss(self,batch_ndx,batch_tup,batch_size,metrics_g):
        input_t, label_t,_series_list, _center_list = batch_tup
        #input_t [batch_size, 1, 32, 48, 48]
        #label_t [batch_size, 2]
        #series_list [batch_size,1]
        #center_list [batch_size, 3]
        input_g = input_t.to(self.device,non_blocking=True)
        label_g = label_t.to(self.device,non_blocking=True)
        #nếu non_blocking=True, khi batch 1 đang lên gpu thì cpu chuẩn bị batch 2 luôn
        logits_g, probability_g = self.model(input_g) #  trả về output logit và output softmax

        #reduction= none là ko gộp loss của các sample trong batch lại, ví dụ batch=32 thì trả về tensor độ dài 32
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g =loss_func(logits_g,label_g[:,1])# chỉ lấy label đúng
        '''
        loss_g = 
        tensor([
            loss_sample_1,
            loss_sample_2,
            ...
            loss_sample_32,
        ])
        '''
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)# xài size chứ ko xài batch_size vì ví dụ batch cuối chỉ có 14 phần tử
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1].detach() # chỉ quan tâm label có nodule
        metrics_g[METRICS_PRED_NDX,start_ndx:end_ndx] = probability_g[:,1].detach() # chỉ quan tâm xác suất có nodule
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()


        #line 238
        return loss_g.mean()
    
    def saveCheckpoint(self, epoch_ndx):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": epoch_ndx,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "totalTrainingSamples_count": self.totalTrainingSamples_count,
        }, self.checkpoint_path)
        log.info(f"Saved checkpoint: {self.checkpoint_path}")

    def doTraining(self,epoch_ndx,train_dl,chunk_ndx= None): # training ở mỗi epoch
        self.model.train()#chuyển sang training mode
        train_dl.dataset.shuffleSamples()# đảo thứ tự bên trong 2 list
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset), # số sample trong dataset, train_dl.dataset là tham chiếu tới train_ds
            device=self.device
        ) # ở gpu để lúc sau gán cho tương thích, nếu ko phải metrics_t[...] = label_g[:, 1].detach().cpu()
        train_progress = tqdm(train_dl,
                              desc = f"E{epoch_ndx} C{chunk_ndx} Training",#ví dụ E1 C1 Training:  35%|███▌| .
                              total=len(train_dl))#tổng số batch
        for batch_ndx, batch_tup in enumerate(train_progress):
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx,#thứ tự batch
                                             batch_tup,
                                             train_dl.batch_size,#truyền vào để tính vị trí ghi metrics
                                             trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()
        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu') # chuyển từ gpu về cpu
        #trả về metrics của epoch này sau khi train
    def doValidation(self,epoch_ndx,val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device = self.device,
            )
            val_progress = tqdm(
                val_dl,
                desc = "E{} validation".format(epoch_ndx),
                total = len(val_dl)
            )
            for batch_ndx, batch_tup in enumerate(val_progress):
                self.computeBatchLoss(
                    batch_ndx,batch_tup,val_dl.batch_size,valMetrics_g
                )
        return valMetrics_g.to('cpu')
    def initTensorboardWriters(self):
        log_dir = "/content/luna_dataset/runs" # run/2026-06-19_15.30.10 ( bỏ thời gian)
        self.trn_writer = SummaryWriter(
            log_dir = log_dir + '-trn_cls' #run/2026-06-19_15.30.10-trn_cls
        )
        self.val_writer= SummaryWriter(
            log_dir = log_dir +"-val_cls"
             #run/2026-06-19_15.30.10-val
        )
    def logMetrics(self,epoch_ndx,mode_str,metrics_t,classificationThreshold=0.5,):
        if self.trn_writer is None:
            self.initTensorboardWriters()
        if mode_str == "trn":
            writer = self.trn_writer
        else:
            writer = self.val_writer

        
        metrics_log  =log_train if mode_str =="trn" else log_val

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold #mask label nhỏ hơn 0.5
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold #mask dự đoán nhỏ hơn 0.5
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask
        neg_count = int(negLabel_mask.sum()) #int chuyển từ tensor sang python int
        pos_count = int(posLabel_mask.sum())

        TrueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())#int chuyển từ tensor sang python int
        TruePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falseNeg_count = pos_count - TruePos_count
        falsePos_count = neg_count - TrueNeg_count
        #precision,recall
        metrics_dict={}
        metrics_dict["loss/all"] = metrics_t[METRICS_LOSS_NDX].mean().item() #tổng loss toàn bộ
        metrics_dict["loss/neg"] = (
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean().item()
            if neg_count > 0
            else 0.0
        ) # tong loss cac label ko phai nodule
        metrics_dict["loss/pos"] = (
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean().item()
            if pos_count > 0
            else 0.0
        )# tong loss cac label la nodule
        sample_count = metrics_t.shape[1]
        metrics_dict["correct/all"] = (
            (pos_correct + neg_correct) / sample_count * 100
            if sample_count > 0
            else 0.0
        ) #accuracy
        metrics_dict["correct/neg"] = (
            neg_correct / neg_count * 100
            if neg_count > 0
            else 0.0
        ) # trong các neg thật sự thì dự đoán đúng bao nhiêu
        #recall lớp neg
        metrics_dict["correct/pos"] = (
            pos_correct / pos_count * 100
            if pos_count > 0
            else 0.0
        ) # trong các pos thật sự thì dự đoán đúng bao nhiêu
        #recall lớp pos

        recall_denominator = TruePos_count + falseNeg_count
        precision_denominator = TruePos_count + falsePos_count

        recall = (
            TruePos_count / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        precision = (
            TruePos_count / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        metrics_dict["pr/recall"] = recall # thực ra giống với correct/pos
        metrics_dict["pr/precision"] = precision
        metrics_dict["pr/f1_score"] = f1_score
        for key, value in metrics_dict.items():
            writer.add_scalar(
                key, # tên biểu đồ
                value, #trục y
                self.totalTrainingSamples_count #trục x
            )

        writer.flush()#writer.flush() dùng để ép các dữ liệu đang nằm trong bộ đệm được ghi ngay xuống file log của TensorBoard.
        metrics_log .info(
            f"E{epoch_ndx} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['correct/all']:-5.1f}% correct"
        )

        metrics_log .info(
            f"E{epoch_ndx} {mode_str + '_neg':8} {metrics_dict['loss/neg']:.4f} loss, "
            f"{metrics_dict['correct/neg']:-5.1f}% correct "
            f"({neg_correct} of {neg_count})"
        )

        metrics_log .info(
            f"E{epoch_ndx} {mode_str + '_pos':8} {metrics_dict['loss/pos']:.4f} loss, "
            f"{metrics_dict['correct/pos']:-5.1f}% correct "
            f"({pos_correct} of {pos_count})"
        )
        metrics_log.info(
            f"E{epoch_ndx} {mode_str:8} "
            f"{metrics_dict['pr/precision']:.4f} precision, "
            f"{metrics_dict['pr/recall']:.4f} recall, "
            f"{metrics_dict['pr/f1_score']:.4f} f1 score"
        )

        metrics_log.info(
            f"E{epoch_ndx} {mode_str:8} "
            f"TP:{TruePos_count} TN:{TrueNeg_count} "
            f"FP:{falsePos_count} FN:{falseNeg_count}"
        )
        
    def main(self):
        val_dl = self.initValDl()

        from dsets import getCandidateInfoToList
        import random

        all_series_uid_list = sorted({
            x.series_uid for x in getCandidateInfoToList()
        })

        try:
            for epoch_ndx in range(self.start_epoch, self.start_epoch + self.cli_args.epochs):
                random.shuffle(all_series_uid_list)

                trnMetrics_list = []

                for chunk_ndx, series_uid_batch in enumerate(self.chunks(all_series_uid_list, 10), 1):
                    log.info(f"E{epoch_ndx} train on {len(series_uid_batch)} CTs")

                    train_dl = self.initTrainDl(series_uid_batch)

                    if train_dl is None:
                        log.info("Skip CT batch because no positive samples.")
                        continue

                    trnMetrics_t = self.doTraining(epoch_ndx, train_dl,chunk_ndx)
                    trnMetrics_list.append(trnMetrics_t)

                if not trnMetrics_list:#Nếu tất cả chunk bị skip thì trnMetrics_list rỗng và torch.cat() sẽ báo lỗi.
                    raise RuntimeError(
                        f"E{epoch_ndx}: Không có chunk training nào chứa positive sample."
                    )

                trnMetrics_t = torch.cat(trnMetrics_list, dim=1)
                self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics_t)

                self.saveCheckpoint(epoch_ndx)

        finally:
            if self.trn_writer is not None:
                self.trn_writer.close()

            if self.val_writer is not None:
                self.val_writer.close()
if __name__ == '__main__':
    settingLogging()
    LunaTrainingApp().main()
