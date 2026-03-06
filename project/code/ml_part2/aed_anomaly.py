# 필요한 모듈 임포트
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Autoencoder 클래스 생성
class AutoEncoder(nn.Module):
    '''완전 연결 Autoencoder 모델'''
    def __init__(self, input_dim: int, hidden_dims = (16, 8)):
        super().__init__()
        h1, h2 = hidden_dims

        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(in_features = input_dim, out_features = h1),
            nn.ReLU(),
            nn.Linear(in_features = h1, out_features = h2),
            nn.ReLU()
        )

        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(in_features = h2, out_features = h1),
            nn.ReLU(),
            nn.Linear(in_features = h1, out_features = input_dim)  # 출력 차원 = 입력 차원
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Autoencoder 기반 이상치 탐지 클래스
class AEAnomalyDetector:
    '''
    - fit(X_train): 정상 데이터로 Autoencoder 학습
    - recon_error(X): 입력 X에 대한 재구성 오차 벡터 반환
    '''
    def __init__(
        self,
        input_dim: int,
        hidden_dims = (16, 8),
        lr: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = AutoEncoder(input_dim = input_dim, hidden_dims = hidden_dims).to(self.device)
        self.criterion = nn.MSELoss(reduction = 'mean')
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr)

    # Tensort 데이터 타입 변환(float32)
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        if not isinstance(X, np.ndarray):
            raise TypeError('X는 numpy.ndarray 여야 합니다.')
        return torch.tensor(data = X, dtype = torch.float32, device = self.device)

    # Autoencoder 신경망 학습
    def fit(self, X: np.ndarray):
        self.model.train()
        
        X_tensor = self._to_tensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset = dataset, batch_size = self.batch_size, shuffle = True)

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch.size(0)

            epoch_loss /= len(dataset)
            if self.verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.num_epochs):
                print(f'Epoch {epoch:3d}/{self.num_epochs}, loss = {epoch_loss:.6f}')

    # X의 샘플별 재구성 오차(MSE) 계산 메서드
    def recon_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_tensor = self._to_tensor(X)

        with torch.no_grad():
            recon = self.model(X_tensor)
            error = torch.mean(input = (X_tensor - recon) ** 2, dim = 1)

        return error.cpu().numpy()