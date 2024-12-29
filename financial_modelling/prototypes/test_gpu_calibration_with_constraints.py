import torch
import logging

class NonLinearModel:
    def __init__(self, initial_params, device = "GPU"):
        self.device = "GPU"
        self.initial_params = initial_params
        pass

    def fit(self, x_train, y_train):
        if self.device != "GPU":
            logging.info("Device is not GPU. Moving data to GPU.")
        x_train = x_train.to("cuda")
        y_train = y_train.to("cuda")
        pass

    def predict(self, x_test):
        pass
    
    @staticmethod
    def functional_form(x, maturity, params):
        a, b, rho, m, sigma = params
        return torch.sqrt(a + b * (rho * (x - m) + torch.sqrt((x - m) ** 2 + sigma ** 2)))/torch.sqrt(maturity)

if __name__ == "main":
    #generate multiple train data so that we can try to fit multiple models at once
    initial_params = [0.05, 0.2, 0.0, 0.0, 0.1]
    model = NonLinearModel(initial_params)
    