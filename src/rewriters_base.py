from abc import ABC, abstractmethod

class RewriteStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_target(self, src: str) -> bool:
        pass

    @abstractmethod
    def rewrite(self, src: str, i: int) -> str:
        pass

    def normalize(self, src: str) -> str:
        return self.rewrite(src, 0)

    @abstractmethod
    def get_current_state(self, src: str) -> int:
        pass