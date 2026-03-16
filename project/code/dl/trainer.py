# 필요한 모듈을 임포트합니다.
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 재현 가능한 결과를 보장하는 시드 고정 함수를 생성합니다.
def set_seed(seed: int = 0, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

# 모델 학습을 위한 Trainer 클래스를 생성합니다.
class Trainer:
    '''
    Trainer는 PyTroch 기반 신경망 모델 학습을 위한 범용 클래스입니다.
    다층 퍼셉트론(MLP), 합성곱 신경망(CNN) 등 다양한 구조의 모델을 
    동일한 학습 루프에서 재사용할 수 있도록 설계되었습니다.
    
    모델의 구조(은닉층, 활성화 함수, 드롭아웃, 배치 정규화 등)를 변경하더라도
    이 클래스를 그대로 재사용할 수 있습니다.
    '''
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, 
                device = None, flatten: bool = True):
        '''
        Trainer 객체를 초기화합니다.
        
        Parameters
        ----------
        model: nn.Module
            학습할 PyTorch 모델 객체입니다.
        criterion: nn.Module
            손실 함수입니다.
        optimizer: torch.optim.Optimizer
            모델 파라미터를 업데이트할 옵티마이저입니다.
        train_loader: DataLoader
            훈련 데이터 로더입니다.
        test_loader: DataLoader
            시험(또는 검증) 데이터 로더입니다.
        device: torch.device, optional
            연산에 사용할 장치(CPU/GPU). None이면 자동으로 선택합니다.
        flatten: bool, default = True
            입력 데이터를 1차원으로 평탄화할지 여부입니다.
            (MLP 사용 시 True, CNN 사용 시 False)
        '''
        
        # 1) 디바이스를 자동 선택합니다. 필요하면 외부에서 지정할 수 있습니다.
        # [참고] 모델에 이미 올라가 있는 실제 디바이스로 설정합니다.
        self.model = model
        
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    device = torch.device('mps')
                else:
                    device = torch.device('cpu')
                
        self.device = device

        # 2) 나머지 구성 요소들을 설정합니다.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.flatten = flatten
        
        # 3) 학습 이력 저장용 빈 리스트를 생성합니다.
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []

        # 현재 설정된 디바이스를 출력합니다.
        print(f'[Trainer] Using device: {self.device}')

    # 미니 배치 이미지와 라벨 데이터를 반환하는 메서드를 생성합니다.
    def prepare_batch(self, images, labels):

        # flatten이 True이면 이미지를 1차원으로 변환하여 MLP 입력 형태로 만듭니다.
        # flatten이 False이면 원래 이미지 형태(CNN 입력)를 유지합니다.
        if self.flatten:
            images = images.reshape(images.shape[0], -1)
        x = images.to(self.device)
        y = labels.to(self.device)
        
        return x, y

    # 미니 배치 입력 텐서와 라벨로 모델을 학습하는 메서드를 생성합니다.
    def train_batch(self, x, y):
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # 에포크 루프를 돌면서 모델을 여러 번 반복 학습합니다.
    def fit(self, n_epochs: int = 10, reset_history: bool = True):
        '''
        모델을 지정한 epoch 수만큼 학습합니다.
        
        Parameters
        ----------
        epochs : int
            학습 epoch 수입니다.
        reset_history : bool, default = True
            True이면 기존 학습 이력을 초기화하고,
            False이면 기존 이력 뒤에 이어서 누적합니다.
        '''
        if reset_history:
            self.train_losses = []
            self.train_accs = []
            self.test_accs = []
        
        for epoch in range(n_epochs):
            self.model.train()
            n_sample, loss_sum = 0, 0.0

            for images, labels in self.train_loader:
                x, y = self.prepare_batch(images, labels)
                batch_size = y.shape[0]
                n_sample += batch_size

                batch_loss = self.train_batch(x, y)
                loss_sum += batch_loss * batch_size

            # 에포크 손실값 평균을 계산합니다.
            train_loss = loss_sum / n_sample

            # 에포크별 훈련셋과 시험셋 정확도를 계산합니다.
            train_acc = self.accuracy(self.train_loader)
            test_acc = self.accuracy(self.test_loader)

            # 학습 이력을 기록합니다.
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)

            print(f'[Epoch {epoch+1:02d}] Loss = {train_loss:.4f}, '
                f'Train_Acc = {train_acc:.4f}, Test_Acc = {test_acc:.4f}')

        # 학습 이력을 history로 생성합니다.
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_accs': self.test_accs,
        }
        
        return history

    # 데이터 로더를 지정하면 예측 확률, 예측값, 실제값을 반환하는 메서드를 생성합니다.
    @torch.no_grad()
    def predict_loader(self, data_loader):
        '''
        주어진 데이터 로더에 대해 (y_prob, y_pred, y_true)를 
        1차원 텐서 형태로 반환합니다.
        '''
        # 모델을 평가 모드로 전환합니다.
        self.model.eval()

        # 전체 예측 확률, 예측값과 실제값을 저장할 빈 리스트를 생성합니다.
        all_prob = []
        all_pred = []
        all_true = []

        # 모델 출력을 생성하고, 최댓값인 클래스를 예측값으로 선택합니다.
        for images, labels in data_loader:
            x, y = self.prepare_batch(images, labels)
            logits = self.model(x)
            y_prob = torch.softmax(input = logits, dim = 1)
            y_pred = y_prob.argmax(dim = 1)
            
            all_prob.append(y_prob.cpu())
            all_pred.append(y_pred.cpu())
            all_true.append(y.cpu())

        # 배치별 텐서를 하나로 연결합니다.
        y_prob = torch.cat(tensors = all_prob, dim = 0)
        y_pred = torch.cat(tensors = all_pred, dim = 0)
        y_true = torch.cat(tensors = all_true, dim = 0)

        return y_prob, y_pred, y_true

    # 정확도를 평가하는 메서드를 생성합니다.
    @torch.no_grad()
    def accuracy(self, data_loader):
        '''
        주어진 데이터 로더에 대해 정확도를 계산합니다.
        '''
        # 모델을 평가 모드로 전환합니다.
        self.model.eval()

        # 전체 샘플 수와 정답 수를 저장할 변수를 생성합니다.
        n_sample = 0
        correct = 0

        # 배치 단위로 예측값을 생성하고 정확도를 누적 계산합니다.
        for images, labels in data_loader:
            x, y = self.prepare_batch(images, labels)
            logits = self.model(x)
            y_pred = logits.argmax(dim = 1)

            n_sample += y.shape[0]
            correct += (y_pred == y).sum().item()

        # 전체 정확도를 계산합니다.
        acc = correct / n_sample
        
        return acc
    
    # 손실 곡선을 시각화하는 메서드를 생성합니다.
    def plot_loss(self):
        '''
        에포크별 훈련 손실값 변화를 시각화합니다.
        '''
        plt.figure(figsize = (4, 4), dpi = 120)
        plt.plot(self.train_losses, label = '훈련셋 손실값')
        plt.xlabel(xlabel = '에포크')
        plt.ylabel(ylabel = '손실값')
        plt.title(label = '신경망 손실값 변화', fontweight = 'bold')
        plt.legend()
        plt.show()

    # 정확도 곡선을 시각화하는 메서드를 생성합니다.
    def plot_accuracy(self):
        '''
        에포크별 훈련/시험 정확도 변화를 시각화합니다.
        '''
        plt.figure(figsize = (4, 4), dpi = 120)
        plt.plot(self.train_accs, label = '훈련셋 정확도')
        plt.plot(self.test_accs, label = '시험셋 정확도')
        plt.xlabel(xlabel = '에포크')
        plt.ylabel(ylabel = '정확도')
        plt.title(label = '신경망 정확도 변화', fontweight = 'bold')
        plt.legend()
        plt.show()

    # 데이터 로더를 지정하면 혼동 행렬을 반환하는 메서드를 생성합니다.
    def get_confusion_matrix(self, data_loader):
        '''
        주어진 데이터 로더에 대해 혼동 행렬을 생성하여 반환합니다.
        '''
        # 예측값과 실제값을 생성합니다.
        _, y_pred, y_true = self.predict_loader(data_loader = data_loader)

        # 예측값과 실제값을 시리즈로 변환합니다.
        y_true = pd.Series(data = y_true.numpy(), name = 'y_true')
        y_pred = pd.Series(data = y_pred.numpy(), name = 'y_pred')

        # 혼동 행렬을 생성합니다.
        confusion_matrix = pd.crosstab(index = y_true, columns = y_pred)

        return confusion_matrix
    
    # 혼동 행렬을 히트맵으로 시각화하는 메서드를 생성합니다.
    def plot_confusion_matrix(self, data_loader, class_names = None):
        '''
        주어진 데이터 로더에 대해 혼동 행렬을 계산하고 히트맵으로 시각화합니다.
        '''
        # 데이터 로더가 없으면 에러를 일으킵니다.
        if data_loader is None:
            raise ValueError('plot_confusion_matrix()에는 data_loader를 반드시 전달해야 합니다.')

        # 타겟 클래스를 data_loader에서 추출합니다.
        if class_names is None and hasattr(data_loader.dataset, 'classes'):
            class_names = data_loader.dataset.classes

        # 혼동 행렬을 생성합니다.
        cm = self.get_confusion_matrix(data_loader = data_loader).copy()

        # 전체 클래스 순서에 맞춰 행과 열을 재정렬합니다.
        if class_names is not None:
            n_class = len(class_names)
            full_index = list(range(n_class))
            cm = cm.reindex(index = full_index, columns = full_index, fill_value = 0)
            cm.index = [class_names[i] for i in cm.index]
            cm.columns = [class_names[i] for i in cm.columns]
        
        # 혼동 행렬을 히트맵으로 시각화합니다.
        plt.figure(figsize = (8, 6), dpi = 120)
        sns.heatmap(data = cm, cmap = 'viridis_r', annot = True, annot_kws = {'size': 8}, fmt = 'd')
        plt.xlabel(xlabel = '예측값')
        plt.ylabel(ylabel = '실제값')
        plt.title(label = '혼동 행렬', fontweight = 'bold')
        plt.show()

    # 정규화된 이미지를 역정규화하는 메서드를 생성합니다.
    def denormalize_image(self, image, mean = None, std = None):
        '''
        정규화된 단일 이미지 텐서(C, H, W)를 역정규화합니다.
        mean, std가 없으면 원본 이미지를 그대로 반환합니다.
        '''
        if mean is None or std is None:
            return image

        mean = torch.tensor(data = mean, dtype = image.dtype, device = image.device).view(-1, 1, 1)
        std = torch.tensor(data = std, dtype = image.dtype, device = image.device).view(-1, 1, 1)
        
        image = image * std + mean
        image = image.clamp(min = 0, max = 1)
        
        return image
        
    # n개의 오분류된 이미지를 시각화하는 메서드를 생성합니다.
    def plot_misclassified(self, data_loader, n: int = 10, class_names = None, mean = None, std = None):
        '''
        주어진 데이터 로더에서 오분류된 이미지 n개를 찾아
        2행 배치로 시각화합니다.
        '''
        # 타겟 클래스를 data_loader에서 추출합니다.
        if class_names is None:
            if hasattr(data_loader.dataset, 'classes'):
                class_names = data_loader.dataset.classes
        
        # 모델을 평가 모드로 전환합니다.
        self.model.eval()

        # 오분류된 데이터를 저장할 빈 리스트를 생성합니다.
        mis_true = []
        mis_pred = []
        mis_images = []
        mis_indices = []

        # 전체 데이터셋에서의 연속 인덱스를 추적합니다.
        sample_idx = 0

        with torch.no_grad():
            for images, labels in data_loader:
                batch_size = labels.shape[0]
                
                x, y = self.prepare_batch(images, labels)
                logits = self.model(x)
                y_pred = logits.argmax(dim = 1)

                # 오분류 여부를 판단합니다.
                mismatch = (y_pred != y)

                # 오분류 데이터가 있으면 배치 내에서 오분류된 위치를 순회합니다.
                if mismatch.any():
                    for b in range(batch_size):
                        if mismatch[b]:
                            mis_true.append(y[b].cpu())
                            mis_pred.append(y_pred[b].cpu())
                            mis_images.append(images[b].cpu())
                            mis_indices.append(sample_idx + b)

                            if len(mis_images) >= n:
                                break

                # 배치 크기를 누적합니다.
                sample_idx += batch_size

                if len(mis_images) >= n:
                    break

        # 오분류된 샘플이 하나도 없는 경우를 처리합니다.
        if len(mis_images) == 0:
            print('오분류된 샘플이 없습니다.')
            return

        # 시각화할 개수를 설정합니다.
        n_show = min(n, len(mis_images))

        # subplot의 행과 열 개수를 결정합니다.
        n_rows = 2 if n_show > 1 else 1
        n_cols = int(np.ceil(n_show / n_rows))

        # n_rows * n_cols 형태의 subplot을 생성합니다.
        fig, axes = plt.subplots(
            nrows = n_rows, 
            ncols = n_cols, 
            figsize = (2 * n_cols, 2 * n_rows), 
            dpi = 120
        )

        # axes를 1차원 배열로 변환합니다.
        axes = np.atleast_1d(axes).flatten()

        # 오분류된 이미지들을 차례대로 그립니다.
        for ax, t, p, img, idx in zip(
            axes, 
            mis_true[:n_show],
            mis_pred[:n_show],
            mis_images[:n_show],
            mis_indices[:n_show]
        ):
            
            # 이미지를 배열로 변환합니다.
            img_np = img.numpy()

            # 채널이 1이면 흑백 이미지로 시각화합니다.
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np.squeeze(axis = 0)
                ax.imshow(X = img_np, cmap = 'gray')

            # 채널이 3이면 컬러 이미지로 시각화합니다.
            elif img_np.ndim == 3 and img_np.shape[0] >= 3:
                img_vis = img[:3].clone()
                img_vis = self.denormalize_image(image = img_vis, mean = mean, std = std)
                img_np = img_vis.numpy().transpose(1, 2, 0)
                ax.imshow(X = img_np)

            # 기타인 경우를 처리합니다.
            else:
                ax.imshow(X = img_np)

            # 실제값과 예측값의 라벨을 생성합니다.
            true_label = class_names[int(t)] if class_names is not None else int(t)
            pred_label = class_names[int(p)] if class_names is not None else int(p)
            
            ax.set_title(label = f'index: {idx}\ntrue: {true_label} | pred: {pred_label}', 
                        fontsize = 8)
            ax.axis('off')

        # 남는 축이 있으면 숨깁니다.
        for ax in axes[n_show:]:
            ax.axis('off')

        # 전체 이미지의 제목을 추가합니다.
        plt.suptitle(t = '오분류된 샘플', fontweight = 'bold')

        # 개별 이미지의 간격을 자동으로 조정하여 제목과 그래프가 겹치지 않도록 합니다.
        plt.tight_layout()
        plt.show()

## End of Document