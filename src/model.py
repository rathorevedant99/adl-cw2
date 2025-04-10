
class GRADCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward(self, x):
        return self.model(x)
    
    def backward(self, grad_output):
        return self.model.backward(grad_output)
    
    def get_grad_cam(self, x, y):
        return self.model.get_grad_cam(x, y)