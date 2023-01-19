from abc import ABC, abstractmethod


__all__ = [
    'ConstantLR',
    'DiminishingLR',
    'MultiStepLR'
]


class LeaningRate(ABC):
    def __init__(self, alpha: float=1e-3) -> None:
        self.alpha = alpha

        self._lr = alpha
        self.n_step = 0
        super(LeaningRate, self).__init__()

    @abstractmethod
    def step() -> None:
        ...
    
    def get_lr(self) -> float:
        return self._lr
    
    def __call__(self) -> float:
        return self.get_lr()
        

class ConstantLR(LeaningRate):
    def __init__(self, alpha: float=1e-3) -> None:
        super(ConstantLR, self).__init__(alpha)
    
    def step(self) -> None:
        self.n_step += 1


class DiminishingLR(LeaningRate):
    def __init__(self, alpha: float=1e-3, p: float=0.5) -> None:
        if not 0.5 <= p <= 1:
            raise ValueError(f'Invaild p parameter: {p}')
        self.p = p
        super(DiminishingLR, self).__init__(alpha)

    def step(self) -> None:
        self.n_step += 1
        self._lr = self.alpha / self.n_step ** self.p


class MultiStepLR(LeaningRate):
    def __init__(self, alpha: float=1e-3, gamma: float=0.9, milestones: list[int]=[]) -> None:
        self.gamma = gamma
        self.milestones = milestones
        super(MultiStepLR, self).__init__(alpha)

    def step(self) -> None:
        self.n_step += 1
        if self.n_step in self.milestones:
            self._lr = self._lr * self.gamma
