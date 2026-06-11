
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
            pin_memory = self.use_cuda,
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
        )
        return val_dl

if __name__ == '__main__':
    LunaTrainingApp().main()