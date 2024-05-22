import torch
from torch import nn, optim
from torch.nn import functional as F

from utils import io
from utils.datasets import NWPDataset
from torch.utils.data import DataLoader
NUM_WORKERS = 64

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, seed, n_split, input_path, case_study, n_thresh):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1)*1.5)
        
        
        (case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny,
        ) = io.get_casestudy_stuff(
            input_path,
            n_split=n_split,
            case_study=case_study,
            ispadded=True,
            seed=seed,
        )
        self.x_train, self.y_train, in_features, out_features = io.load_data(
            input_path,
            train_dates,
            case_study_max,
            indices_one,
            indices_zero,
            available_models,
        )
        train_dataset = NWPDataset(
            (
                torch.from_numpy(self.x_train),
                torch.from_numpy(self.y_train).unsqueeze(1),
                torch.from_numpy(train_dates),
            )
        )
        self.train_dataloader = DataLoader( train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
        
        mask = torch.from_numpy(mask).float().cuda()
        
        thresh = [
            5 / case_study_max,
            10 / case_study_max,
            20 / case_study_max,
            50 / case_study_max,
            100 / case_study_max,
            150 / case_study_max,
            1 / case_study_max,
        ]
        self.thresholds = thresh[:n_thresh]

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.BCEWithLogitsLoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in self.train_dataloader:
                input, label, date = batch["x"], batch["y"], batch.get("ev_date")
                input = input.cuda()
                date=date.cuda()
                logits, l = self.model(input, date)
                logits_list.append(logits)
                y_p = []
                for i in range(len(self.thresholds)):
                    y_p.append(label.gt(self.thresholds[i]).float())
                y_p = torch.cat(y_p, dim=1)
                labels_list.append(y_p)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f' % (after_temperature_nll))
        
        return self.temperature