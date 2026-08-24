from abc import ABC,abstractmethod
class AIProvider(ABC):
 @abstractmethod
 def complete(self,system,user): raise NotImplementedError
