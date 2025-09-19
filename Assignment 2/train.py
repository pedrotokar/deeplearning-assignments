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


class YOLOv3Loss(nn.Module):
    # (N, B, (tx,ty,tw,th,t0), S, S)

    # pred, true tensor
    # (N, B, S, S, 5+C)
    # N: batch size
    # B: n anchors
    # SxS: grid size
    # 5+C [p0, x, y, w, h, p1,...,pc]

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        self.lambda_noobj = 1
        self.lambda_coord = 1

        self.lambda_class = 1
        self.lambda_obj = 1


    def compute_noobj_loss(self, pred, true, noobj_mask):
        loss = self.bce(
            pred[..., 0:1][noobj_mask],
            true[..., 0:1][noobj_mask],
        )

        return loss

    def compute_obj_loss(self, pred, true, obj_mask):
        loss = self.bce(
            pred[..., 0:1][obj_mask],
            true[..., 0:1][obj_mask]
        )

        return loss

    def compute_coord_loss(self, pred, true, obj_mask, anchors):
        box_pred = pred[..., 1:5][obj_mask]
        box_true = true[..., 1:5][obj_mask]

        wh_pred = box_pred[..., 2:4]
        wh_true = box_true[..., 2:4]

        N, B, S, _ = true.shape[:4]

        anchors_new= anchors.reshape(1, B, 1, 1, 2).expand(N, B, S, S, 2).to(true.device)
        anchor_priors = anchors_new[obj_mask]

        loss_xy = self.mse(
            self.sigmoid(box_pred[..., :2]), box_true[..., :2]
        )

        loss_wh = self.mse(
            wh_pred, torch.log(wh_true/anchor_priors)
        )

        return loss_xy + loss_wh


def compute_class_loss(self, pred, true, obj_mask):
        obj_pred = pred[obj_mask]
        obj_true = true[obj_mask]

        pred_class = obj_pred[..., 5:]
        true_indices = obj_true[..., 5].long()

        true_class = torch.zeros_like(pred_class)
        true_class.scatter_(1, true_indices.unsqueeze(1), 1)

        loss = self.bce(pred_class, true_class)

        return loss

    def forward(self, pred, true, anchors):
        obj_mask = true[..., 0] == 1
        noobj_mask = ~obj_mask

        obj_loss = self.compute_obj_loss(pred, true, obj_mask)
        noobj_loss = self.compute_noobj_loss(pred, true, noobj_mask)
        coord_loss = self.compute_coord_loss(pred, true, obj_mask, anchors)
        class_loss = self.compute_class_loss(pred, true, obj_mask)

        final_loss = self.lambda_noobj * noobj_loss + \
                     self.lambda_coord * coord_loss + \
                     obj_loss + \
                     class_loss

        return final_loss
