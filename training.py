
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
METRICS_LABEL_NDX = 0 # label thật các sample
METRICS_PRED_NDX = 1 # xác suất label dự đoán các sample 
METRICS_LOSS_NDX = 2 # loss của các sample
METRICS_SIZE = 3
log = logging.getLogger(__name__)
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
        #line63
        
        self.cli_args = parser.parse_args(sys_argv) # nếu tham số là --help thì in ra rồi dừng luôn
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model= self.initModel()
        self.optimizer = self.initOptimizer()
        self.totalTrainingSamples_count = 0
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
    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride = 10,
            isValSet_bool = False
        )
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
    def doTraining(self,epoch_ndx,train_dl): # training ở mỗi epoch
        self.model.train()#chuyển sang training mode
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset), # số sample trong dataset, train_dl.dataset là tham chiếu tới train_ds
            device=self.device
        ) # ở gpu để lúc sau gán cho tương thích, nếu ko phải metrics_t[...] = label_g[:, 1].detach().cpu()
        train_progress = tqdm(train_dl,
                              desc="E{} Training".format(epoch_ndx),#ví dụ E1 Training:  35%|███▌| .
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
    def logMetrics(self,epoch_ndx,mode_str,metrics_t,classificationThreshold=0.5,):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold #mask label nhỏ hơn 0.5
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold #mask dự đoán nhỏ hơn 0.5
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask
        neg_count = int(negLabel_mask.sum()) #int chuyển từ tensor sang python int
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())#int chuyển từ tensor sang python int
        pos_correct = int((posLabel_mask & posPred_mask).sum())
        metrics_dict={}
        metrics_dict["loss/all"] = metrics_t[METRICS_LOSS_NDX].mean() #tổng loss toàn bộ
        metrics_dict["loss/neg"] = metrics_t[METRICS_LOSS_NDX,negLabel_mask].mean() # tong loss cac label ko phai nodule
        metrics_dict["loss/pos"] = metrics_t[METRICS_LOSS_NDX,posLabel_mask].mean()# tong loss cac label la nodule

        metrics_dict["correct/all"] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100 #accuracy
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100 # trong các neg thật sự thì dự đoán đúng bao nhiêu
        #recall lớp neg
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100 # trong các pos thật sự thì dự đoán đúng bao nhiêu
        #recall lớp pos

        log.info(
            f"E{epoch_ndx} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['correct/all']:-5.1f}% correct"
        )

        log.info(
            f"E{epoch_ndx} {mode_str + '_neg':8} {metrics_dict['loss/neg']:.4f} loss, "
            f"{metrics_dict['correct/neg']:-5.1f}% correct "
            f"({neg_correct} of {neg_count})"
        )

        log.info(
            f"E{epoch_ndx} {mode_str + '_pos':8} {metrics_dict['loss/pos']:.4f} loss, "
            f"{metrics_dict['correct/pos']:-5.1f}% correct "
            f"({pos_correct} of {pos_count})"
        )
        
    def main(self):
        train_dl = self.initTrainDl() # một DataLoader
        val_dl = self.initValDl() # một dataloader
        for epoch_ndx in range(1,self.cli_args.epochs +1):
            trnMetrics_t = self.doTraining(epoch_ndx,train_dl)
            self.logMetrics(epoch_ndx,'trn',trnMetrics_t)

            valMetrics_t= self.doValidation(epoch_ndx,val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)
        #train_dl.dataset giữ tham chiếu đến dataset gốc là train_ds
        #train_dl.daset tương đương train_ds
if __name__ == '__main__':
    LunaTrainingApp().main()
