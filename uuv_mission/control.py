# control.py

class PDController:
    def __init__(self, kp=0.15, kd=0.6):
        
        self.kp = kp
        self.kd = kd
        self.prev_error = 0  

    def control(self, reference: float, output: float) -> float:
        
        error = reference - output
        
        
        control_action = self.kp * error + self.kd * (error - self.prev_error)
        
       
        self.prev_error = error
        
        return control_action