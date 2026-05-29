from dataclasses import dataclass


@dataclass
class LevelStorage:
    pass


@dataclass
class DenseStorage(LevelStorage):
    size: int


@dataclass
class CompressedStorage(LevelStorage):
    pos: object
    crd: object


@dataclass
class SingletonStorage(LevelStorage):
    crd: object
