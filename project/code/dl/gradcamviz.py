# 필요한 모듈을 임포트합니다.
import torch
from torch import Tensor
from torchcam.methods import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Grad-CAM 히트맵을 계산하고 시각화하는 클래스를 생성합니다.
class GradCAMViz:
    '''
    단일 이미지 또는 데이터셋 표본에 대해 Grad-CAM 히트맵을 계산하고 시각화하는 클래스입니다.

    Parameters
    ----------
    model: torch.nn.Module
        학습이 완료된 이미지 분류 모델입니다.
        
    target_layer: torch.nn.Module | str
        Grad-CAM을 계산할 타겟 레이어입니다.
        torchcam의 GradCAM 클래스에 그대로 전달됩니다.
        
    data_loader : torch.utils.data.DataLoader
        Grad-CAM 시각화에 사용할 데이터 로더입니다.
        내부적으로 index에 해당하는 이미지와 라벨을 추출할 때 사용됩니다.
        data_loader.dataset.classes 속성이 존재하면 클래스 이름을 자동으로 사용합니다.
        
    device: str | torch.device | None, default = None
        모델 연산 장치입니다. None이면 모델 파라미터의 device를 자동으로 사용합니다.
    '''
    # 초기 설정과 Grad-CAM 객체 생성을 처리하는 메서드를 정의합니다.
    def __init__(self, model, target_layer, data_loader, device = None):
        self.model = model
        self.model.eval()

        # [참고] 모델에 이미 올라가 있는 실제 디바이스로 설정합니다.
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
        
        self.device = torch.device(device)
        self.data_loader = data_loader
        self.dataset = data_loader.dataset
        self.target_layer = target_layer
        self.gradcam = GradCAM(model = self.model, target_layer = self.target_layer)

    # -------------------------------------------------
    # 내부 보조 메서드
    # -------------------------------------------------
    
    # 라벨 값을 파이썬 정수형으로 변환하는 메서드를 정의합니다.
    @staticmethod
    def _to_numpy_label(label):
        if isinstance(label, Tensor):
            return int(label.item())
        return int(label)

    # 이미지 텐서의 자료형과 차원을 검증하는 메서드를 정의합니다.
    @staticmethod
    def _validate_image_tensor(image: Tensor):
        if not isinstance(image, Tensor):
            raise TypeError('image는 torch.Tensor여야 합니다.')
        if image.ndim != 3:
            raise ValueError('image는 (C, H, W) 형태의 3차원 텐서여야 합니다.')

    # Grad-CAM 히트맵을 0~1 범위로 정규화하는 메서드를 정의합니다.
    @staticmethod
    def _normalize_cam(cam):
        cam = cam.detach().squeeze().float().cpu()
        cam_min = cam.min()
        cam_max = cam.max()
        
        if torch.isclose(cam_max, cam_min):
            return np.zeros(tuple(cam.shape), dtype = np.float32)
        
        cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.numpy()

    # 평균과 표준편차를 채널 수에 맞게 브로드캐스팅하는 메서드를 정의합니다.
    @staticmethod
    def _broadcast_stats(
        value,
        channels: int,
        name: str,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if value is None:
            return None

        if isinstance(value, (int, float)):
            arr = [float(value)] * channels
        else:
            arr = list(value)
            if len(arr) != channels:
                raise ValueError(
                    f'{name}의 길이는 이미지 채널 수({channels})와 같아야 합니다. '
                    f'현재 {name} 길이: {len(arr)}'
                )
        
        return torch.tensor(data = arr, dtype = dtype, device = device).view(channels, 1, 1)

    # 클래스 이름 목록을 우선순위에 따라 결정하는 메서드를 정의합니다.
    def _resolve_class_names(self):
        if hasattr(self.dataset, 'classes'):
            return self.dataset.classes
        
        return None

    # 클래스 인덱스를 클래스 이름 문자열로 변환하는 메서드를 정의합니다.
    def _class_name(self, class_idx: int):
        class_names = self._resolve_class_names()
        
        if class_names is None:
            return str(class_idx)
        
        return str(class_names[class_idx])

    # -------------------------------------------------
    # 데이터 추출 및 예측
    # -------------------------------------------------

    # 데이터셋에서 지정한 표본의 이미지와 라벨을 추출하는 메서드를 정의합니다.
    def get_sample(self, index: int):
        '''
        data_loader 기준 인덱스로 (image, label)을 반환합니다.
        transform이 적용된 이미지를 반환합니다.
        '''
        running = 0

        for images, labels in self.data_loader:
            batch_size = images.shape[0]
    
            if running + batch_size > index:
                local_idx = index - running
                image = images[local_idx]
                label = labels[local_idx]
    
                self._validate_image_tensor(image)
                label = self._to_numpy_label(label)
    
                return image, label
    
            running += batch_size
    
        raise IndexError('index가 data_loader 범위를 초과했습니다.')

    # 단일 이미지에 대한 로짓과 예측 클래스를 반환하는 메서드를 정의합니다.
    def predict_image(self, image: Tensor):
        '''
        단일 이미지 텐서 (C, H, W)를 입력받아 로짓과 예측 클래스를 반환합니다.
        '''
        self._validate_image_tensor(image)
        image = image.to(self.device)
        logits = self.model(image.unsqueeze(dim = 0))
        pred = int(logits.argmax(dim = 1).item())
        
        return logits, pred

    # 데이터셋 표본의 로짓, 예측값, 실제값과 이미지를 반환하는 메서드를 정의합니다.
    def get_logits_pred_true(self, index: int):
        '''
        데이터셋 인덱스를 지정하여 (logits, pred, true, image)를 반환합니다.
        '''
        image, true = self.get_sample(index = index)
        logits, pred = self.predict_image(image = image)
        
        return logits, pred, true, image

    # -------------------------------------------------
    # 역정규화 및 시각화용 이미지 변환
    # -------------------------------------------------
    
    # 정규화된 이미지를 원래 스케일로 되돌리는 메서드를 정의합니다.
    def denormalize(self, image: Tensor, mean = None, std = None):
        '''
        정규화된 이미지를 원래 스케일로 되돌립니다.
        mean/std가 없으면 원본을 그대로 반환합니다.
        '''
        self._validate_image_tensor(image)

        if mean is None or std is None:
            return image.detach().clone()

        channels = image.shape[0]
        
        mean = self._broadcast_stats(
            value = mean,
            channels = channels,
            name = 'mean',
            dtype = image.dtype,
            device = image.device,
        )
        
        std = self._broadcast_stats(
            value = std,
            channels = channels,
            name = 'std',
            dtype = image.dtype,
            device = image.device,
        )
        
        return image * std + mean

    # 이미지 텐서를 matplotlib 출력용 numpy 배열로 변환하는 메서드를 정의합니다.
    def image_to_display_array(self, image: Tensor, denorm: bool = True, mean = None, std = None):
        '''
        matplotlib 시각화를 위해 이미지 텐서를 numpy 배열로 변환합니다.

        Returns
        -------
        np.ndarray
            흑백 이미지는 (H, W), 컬러 이미지는 (H, W, C) 형태로 반환합니다.
        '''
        self._validate_image_tensor(image)

        image = image.detach().cpu().float().clone()
        
        if denorm:
            image = self.denormalize(image = image, mean = mean, std = std)
            image = image.detach().cpu().float()

        if image.shape[0] == 1:
            array = image.squeeze(0).numpy()
            array = np.clip(a = array, a_min = 0.0, a_max = 1.0)
        else:
            array = image.permute(1, 2, 0).numpy()
            array = np.clip(a = array, a_min = 0.0, a_max = 1.0)
            
        return array

    # -------------------------------------------------
    # Grad-CAM 계산
    # -------------------------------------------------
        
    # 지정한 클래스 기준으로 Grad-CAM 히트맵을 계산하는 메서드를 정의합니다.
    def compute_cam(self, image: Tensor, class_idx: int):
        '''
        지정한 클래스 기준 Grad-CAM 히트맵을 계산합니다.

        Notes
        -----
        동일한 logits로 두 번 역전파하지 않기 위해, 호출할 때마다 로짓을 새로 계산합니다.
        '''
        self._validate_image_tensor(image)

        image = image.to(self.device)
        logits = self.model(image.unsqueeze(dim = 0))
        cam = self.gradcam(class_idx = class_idx, scores = logits)
        
        return self._normalize_cam(cam[0])

    # 예측 클래스와 실제 클래스 기준 Grad-CAM을 함께 계산하는 메서드를 정의합니다.
    def compute_pred_true_cam(self, index: int):
        '''
        하나의 표본에 대해 예측 클래스 기준/실제 클래스 기준 Grad-CAM을 모두 계산합니다.
        '''
        image, true = self.get_sample(index = index)
        logits, pred = self.predict_image(image = image)

        pred_cam = self.compute_cam(image = image, class_idx = pred)
        true_cam = self.compute_cam(image = image, class_idx = true)

        return {
            'image': image,
            'true': true,
            'pred': pred,
            'logits': logits.detach().cpu(),
            'pred_cam': pred_cam,
            'true_cam': true_cam,
            'true_name': self._class_name(true),
            'pred_name': self._class_name(pred),
        }

    # -------------------------------------------------
    # 히트맵 시각화
    # -------------------------------------------------
    
    # 단일 표본의 원본 이미지만 시각화하는 메서드를 정의합니다.
    def plot_original_image(
        self,
        index: int,
        denorm: bool = True,
        mean = None,
        std = None,
        ax = None,
        title: bool = True,
    ):
        '''
        단일 표본의 원본 이미지만 시각화합니다.

        Parameters
        ----------
        index : int
            data_loader 기준 표본 인덱스입니다.

        denorm : bool, default = True
            True이면 mean, std를 사용해 역정규화한 뒤 시각화합니다.

        mean, std : float | list | tuple | None
            역정규화에 사용할 평균과 표준편차입니다.

        ax : matplotlib.axes.Axes | None, default = None
            None이면 현재 axes를 사용합니다.

        title : bool, default = True
            True이면 index, true, pred 정보를 제목에 표시합니다.
        '''
        image, true = self.get_sample(index = index)
        _, pred = self.predict_image(image = image)

        if ax is None:
            ax = plt.gca()

        display_image = self.image_to_display_array(
            image = image, 
            denorm = denorm, 
            mean = mean, 
            std = std,
        )

        if image.shape[0] == 1:
            ax.imshow(X = display_image, cmap = 'gray')
        else:
            ax.imshow(X = display_image)

        if title:
            true_name = self._class_name(true)
            pred_name = self._class_name(pred)
            ax.set_title(
                label = f'index: {index}\ntrue: {true_name} | pred: {pred_name}', 
                fontsize = 8
            )

        ax.axis('off')

        return ax

    # 단일 표본의 Grad-CAM 결과를 1개 축에 시각화하는 메서드를 정의합니다.
    def plot_grad_cam(
        self,
        index: int,
        class_mode: str = 'pred',
        style: str = 'simple',
        overlay: bool = True,
        cmap: str = 'jet',
        alpha: float = 0.5,
        gamma: float = 2.0,
        cam_threshold: float = 0.3,
        interpolation: str = 'bilinear',
        denorm: bool = True,
        mean = None,
        std = None,
        ax = None,
        title: bool = True,
    ):
        '''
        단일 표본에 대한 Grad-CAM 결과를 하나의 축에 시각화합니다.

        Parameters
        ----------
        class_mode : {'pred', 'true'}
            'pred'이면 예측 클래스 기준, 'true'이면 실제 클래스 기준으로 Grad-CAM을 계산합니다.
            
        style : {'simple', 'enhanced'}, default = 'simple'
            'simple'이면 원본 이미지 위에 CAM을 그대로 겹쳐 그립니다.
            'enhanced'이면 gamma와 threshold를 적용한 뒤 CAM을 겹쳐 그립니다.
        
        overlay : bool, default = True
            True이면 원본 이미지 위에 히트맵을 겹쳐 그립니다.
            False이면 히트맵만 그립니다.
        '''
        if class_mode not in {'pred', 'true'}:
            raise ValueError('class_mode는 pred 또는 true만 가능합니다.')

        if style not in {'simple', 'enhanced'}:
            raise ValueError("style은 'simple' 또는 'enhanced'만 가능합니다.")

        image, true = self.get_sample(index = index)
        _, pred = self.predict_image(image = image)

        target_idx = pred if class_mode == 'pred' else true
        cam = self.compute_cam(image = image, class_idx = target_idx)

        if ax is None:
            ax = plt.gca()

        display_image = self.image_to_display_array(
            image = image, 
            denorm = denorm, 
            mean = mean, 
            std = std
        )

        if cam.shape != display_image.shape[:2]:
            h, w = display_image.shape[:2]
            cam = cv2.resize(src = cam, dsize = (w, h), interpolation = cv2.INTER_CUBIC)
        
        if overlay:
            if image.shape[0] == 1:
                ax.imshow(X = display_image, cmap = 'gray')
            else:
                ax.imshow(X = display_image)

        if image.shape[0] == 1:
            cam_to_show = cam
        else:
            if style == 'simple':
                cam_to_show = cam
            else:
                cam_to_show = np.clip(a = cam, a_min = 0.0, a_max = 1.0)
                cam_to_show = cam_to_show ** gamma
                cam_to_show = np.ma.masked_where(cam_to_show < cam_threshold, cam_to_show)

        if overlay:
            ax.imshow(
                X = cam_to_show, 
                cmap = cmap, 
                alpha = alpha, 
                interpolation = interpolation
            )
        else:
            ax.imshow(
                X = cam_to_show, 
                cmap = cmap, 
                interpolation = interpolation
            )
        
        if title:
            true_name = self._class_name(true)
            pred_name = self._class_name(pred)
            mode_name = 'Pred-class' if class_mode == 'pred' else 'True-class'
            ax.set_title(
                label = f'{mode_name} Grad-CAM\nindex: {index} | true: {true_name} | pred: {pred_name}', 
                fontsize = 8
            )
        ax.axis('off')

        return ax

    # 예측 클래스와 실제 클래스 기준 Grad-CAM을 비교 시각화하는 메서드를 정의합니다.
    def plot_pred_true_comparison(
        self,
        index: int,
        style: str = 'simple',
        overlay: bool = True,
        cmap: str = 'jet',
        alpha: float = 0.5,
        gamma: float = 2.0,
        cam_threshold: float = 0.3,
        interpolation: str = 'bilinear',
        denorm: bool = True,
        mean = None,
        std = None,
        figsize = (10, 4),
    ):
        '''
        예측 클래스 기준과 실제 클래스 기준 Grad-CAM을 나란히 시각화합니다.
        '''
        if style not in {'simple', 'enhanced'}:
            raise ValueError("style은 'simple' 또는 'enhanced'만 가능합니다.")
        
        result = self.compute_pred_true_cam(index = index)
        
        display_image = self.image_to_display_array(
            image = result['image'], 
            denorm = denorm, 
            mean = mean, 
            std = std
        )
        
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)

        for ax, cam_key, title in zip(
            axes,
            ['pred_cam', 'true_cam'],
            ['Pred-class Grad-CAM', 'True-class Grad-CAM'],
        ):
            cam = result[cam_key]

            if cam.shape != display_image.shape[:2]:
                h, w = display_image.shape[:2]
                cam = cv2.resize(src = cam, dsize = (w, h), interpolation = cv2.INTER_CUBIC)
            
            if overlay:
                if result['image'].shape[0] == 1:
                    ax.imshow(X = display_image, cmap = 'gray')
                else:
                    ax.imshow(X = display_image)

            if result['image'].shape[0] == 1:
                cam_to_show = cam
            else:
                if style == 'simple':
                    cam_to_show = cam
                else:
                    cam_to_show = np.clip(a = cam, a_min = 0.0, a_max = 1.0)
                    cam_to_show = cam_to_show ** gamma
                    cam_to_show = np.ma.masked_where(cam_to_show < cam_threshold, cam_to_show)

            if overlay:
                ax.imshow(
                    X = cam_to_show, 
                    cmap = cmap, 
                    alpha = alpha, 
                    interpolation = interpolation
                )
            else:
                ax.imshow(
                    X = cam_to_show, 
                    cmap = cmap, 
                    interpolation = interpolation
                )
                
            ax.set_title(
                label = f"{title}\nindex: {index} | true: {result['true_name']} | pred: {result['pred_name']}", 
                fontsize = 8
            )
            ax.axis('off')

        plt.tight_layout()
        
        return fig, axes

    # 단일 표본의 Grad-CAM 결과를 시각화하는 메서드를 정의합니다.
    def plot_sample(
        self,
        index: int,
        class_mode: str = 'pred',
        style: str = 'simple',
        overlay: bool = True,
        cmap: str = 'jet',
        alpha: float = 0.5,
        gamma: float = 2.0,
        cam_threshold: float = 0.3,
        interpolation: str = 'bilinear',
        denorm: bool = True,
        mean = None,
        std = None,
        figsize = (10, 4),
    ):
        '''
        단일 표본에 대해 왼쪽에는 원본 이미지, 오른쪽에는 Grad-CAM 결과를 시각화합니다.

        Parameters
        ----------
        class_mode: {'pred', 'true'}
            'pred'이면 예측 클래스 기준, 'true'이면 실제 클래스 기준으로 Grad-CAM을 계산합니다.
            
        overlay: bool, default = True
            True이면 오른쪽 패널에 원본 이미지 위에 히트맵을 겹쳐 그립니다.
            False이면 오른쪽 패널에 히트맵만 그립니다.
        '''
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        
        self.plot_original_image(
            index = index, 
            denorm = denorm, 
            mean = mean, 
            std = std, 
            ax = axes[0], 
            title = True
        )
        
        self.plot_grad_cam(
            index = index, 
            class_mode = class_mode, 
            style = style, 
            overlay = overlay, 
            cmap = cmap, 
            alpha = alpha, 
            gamma = gamma, 
            cam_threshold = cam_threshold, 
            interpolation = interpolation, 
            denorm = denorm, 
            mean = mean, 
            std = std, 
            ax = axes[1], 
            title = True
        )
        
        plt.tight_layout()
        
        return fig, axes

    # 여러 표본의 Grad-CAM 결과를 격자 형태로 시각화하는 메서드를 정의합니다.
    def plot_multiple_samples(
        self,
        indices,
        nrows: int,
        ncols: int,
        class_mode: str = 'pred',
        style: str = 'simple',
        overlay: bool = True,
        cmap: str = 'jet',
        alpha: float = 0.5,
        gamma: float = 2.0,
        cam_threshold: float = 0.3,
        interpolation: str = 'bilinear',
        denorm: bool = True,
        mean = None,
        std = None,
        figsize = (12, 6),
    ):
        '''
        여러 표본에 대한 Grad-CAM을 격자 형태로 시각화합니다.
        '''
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
        axes = np.array(object = axes).reshape(-1)

        for ax, idx in zip(axes, indices):
            self.plot_grad_cam(
                index = int(idx),
                class_mode = class_mode,
                style = style,
                overlay = overlay,
                cmap = cmap,
                alpha = alpha,
                gamma = gamma, 
                cam_threshold = cam_threshold,
                interpolation = interpolation,
                denorm = denorm,
                mean = mean,
                std = std,
                ax = ax,
                title = True
            )

        for ax in axes[len(indices):]:
            ax.axis('off')

        plt.tight_layout()
        
        return fig, axes
    
    # torchcam hook을 정리하는 메서드를 정의합니다.
    # [참고] torchcam의 GradCAM 객체는 타겟 레이어에 hook을 등록하므로 
    # 이 객체를 여러 번 생성하면 같은 레이어에 hook이 계속 추가됩니다.
    # 따라서 Grad-CAM 시각화 후에 hook을 제거하는 것이 좋습니다.
    def close(self):
        '''
        torchcam hook 정리를 위해 호출합니다.
        '''
        if hasattr(self.gradcam, 'remove_hooks'):
            self.gradcam.remove_hooks()

## End of Document