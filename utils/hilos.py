import threading

class Hilos():
    listhilos = []
    def start(self, funcion, **arg):
        thread = threading.Thread(target=funcion, args=arg)
        thread.start()  # Iniciar el hilo
        self.listhilos.append(thread)
        return thread
    
    def run(self, thread):
        thread.run()  # Ejecutar el hilo
    
    def status(self, thread):
        return thread.is_alive()  # Verificar si el hilo está en ejecución
    
    def stop(self, thread):
        thread._stop()  # Detener el hilo
    def get_threads(self):
        threads = []
        for thread in self.listhilos:
            if self.status(thread):
                threads.append(thread)
        return threads
        
