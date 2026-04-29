#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
When --mu > 0, FedProx proximal term regularization is enabled.
"""

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.client import BaseClient


class FedAvgClient(BaseClient):
    """
    When args.mu > 0, add FedProx proximal term (μ/2)‖w - w_global‖² to the loss.
    """

    def local_train(self, global_weights, global_round):
        """
        Args:
            global_weights: Global model state_dict
            global_round: Current global round
        """
        model = self._build_local_model(global_weights)
        model.train()

        optimizer = self._make_optimizer(model)
        self._adjust_lr(optimizer, global_round)

        mu = getattr(self.args, 'mu', 0.0)
        global_params = None
        if mu > 0:
            global_params = {k: v.clone().detach().to(self.device)
                             for k, v in global_weights.items()}

        epoch_loss = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if outputs.dim() == 3:  # (batch, seq_len, vocab_size)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)),
                                          labels.reshape(-1))
                else:
                    loss = self.criterion(outputs, labels)

                # FedProx proximal term (skipped when mu=0)
                if global_params is not None:
                    prox = sum(
                        torch.sum((param - global_params[name]) ** 2)
                        for name, param in model.named_parameters()
                        if name in global_params
                    )
                    loss = loss + (mu / 2) * prox

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        train_acc = self.evaluate_on_training_data(model)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), train_acc

    # Compatibility fallback: some environments may run with an older BaseClient.
    def evaluate_on_training_data(self, model):
        model.to(self.device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                # NLP sequence tasks: flatten tensors
                if outputs.dim() == 3:
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    labels = labels.reshape(-1)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def _build_local_model(self, global_weights):
        from models.cnn import CNNMnist, CNNCifar, CNNFemnist
        from models.resnet import ResNet18Fed
        from models.rnn import CharLSTM

        if self.args.model == 'cnn':
            if self.args.dataset == 'mnist':
                model = CNNMnist(num_classes=self.args.num_classes)
            elif self.args.dataset == 'femnist':
                model = CNNFemnist(num_classes=self.args.num_classes)
            else:
                model = CNNCifar(num_classes=self.args.num_classes)
        elif self.args.model == 'resnet18':
            norm_type = 'groupnorm' if not self.args.iid else 'groupnorm'  # Unified use of GroupNorm
            in_channels = 1 if self.args.dataset in ('mnist', 'femnist') else 3
            model = ResNet18Fed(num_classes=self.args.num_classes, norm_type=norm_type, in_channels=in_channels)
        elif self.args.model == 'lstm':
            model = CharLSTM(vocab_size=self.args.vocab_size,
                             embed_dim=self.args.embed_dim,
                             hidden_dim=self.args.hidden_dim)
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")

        model.load_state_dict(global_weights)
        model.to(self.device)
        return model