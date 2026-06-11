import logging
log =  logging.getLogger(_name_)

def run(app_cls, *argv):#run(LunaTrainingApp, '--epochs=1')
    #app_cls là class app tôi muốn chạy, *argv là các tham số truyền thêm
    argv = list(argv)#biến tuple thành list argv =['--epochs=1']
    argv.insert(0, '--num-workers=4') #argv = ['--num-workers=4', '--epochs=1']

    log.info(f"Running: {app_cls.__name__}({argv!r}).main()")
    #Running: LunaTrainingApp(['--num-workers=4', '--epochs=1']).main()
    
    app_cls(argv).main()
    log.info(f"Finished: {app_cls.__name__}({argv!r}).main()")