class Trainer():
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.device = device
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training")
        for X, y in progress_bar:
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X.size(0)
            
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating")
            for X, y in progress_bar:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, y)

                running_loss += loss.item() * X.size(0)

                progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

        return self.history