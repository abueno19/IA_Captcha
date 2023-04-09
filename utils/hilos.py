# En esta clase vamos a crear una clase para crear hilos, y poder ejecutar varias tareas a la vez
import threading

class hilos():
    def __init__(self,funcion,**kwargs):
        self.funcion=funcion
        self.arg=kwargs
    def start(self):
        self.thread=threading.Thread(target=self.funcion,args=self.arg)
        self.thread.start()
    def status(self):
        return self.thread.is_alive()
    def stop(self):
        self.thread.join()
        
    
        