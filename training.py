
import sys
import argparse
import datetime

class LunaTrainingApp:
    def __init__(self, sys_argv = None):
        #python -m p2ch13.training --num-workers=4 --epochs=1
        if sys_argv is None:
        #nếu người dùng ko đưa vào argv thì lấy argv từ command line
            sys_argv = sys.arg #[--num-workers=4 , --epochs=1]
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help = 'Number of worker processes for background data loading',
                            default =8,
                            type=int)#khi thấy arg num worker thì thấy giá trị int ngay sao nó
        parser.add_argument('--batch-size', help='batch size used for training',default=32,type=int)
        parser.add_argument('--epochs',help='Number of epochs to train for',default=1,type=int,)
        #line63

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
if __name__ == '__main__':
    LunaTrainingApp().main()