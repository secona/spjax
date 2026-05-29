from spjax.levels import (
    CompressedSpec,
    DenseSpec,
    LevelSpec,
    SingletonSpec,
)


class SparseIterator:
    def valid(self) -> bool: ...

    def coord(self) -> int: ...

    def pos(self) -> int: ...

    def next(self) -> None: ...

    def seek(self, coord: int) -> None: ...


class DenseCoordinateIterator(SparseIterator):
    def __init__(self, size: int):
        self.i = 0
        self.size = size

    def valid(self) -> bool:
        return self.i < self.size

    def coord(self) -> int:
        return self.i

    def pos(self) -> int:
        return self.i

    def next(self) -> None:
        self.i += 1

    def seek(self, coord: int) -> None:
        self.i = coord


class CompressedIterator(SparseIterator):
    def __init__(self, crd, p_begin: int, p_end: int):
        self.crd_arr = crd
        self.p = p_begin
        self.p_end = p_end

    def valid(self) -> bool:
        return self.p < self.p_end

    def coord(self) -> int:
        return int(self.crd_arr[self.p])

    def pos(self) -> int:
        return self.p

    def next(self) -> None:
        self.p += 1

    def seek(self, coord: int) -> None:
        while self.valid() and self.coord() < coord:
            self.next()


class IteratorFactory:
    @staticmethod
    def make_iterator(
        spec: LevelSpec,
        storage,
        *,
        parent_pos: int | None = None,
    ) -> SparseIterator:
        if isinstance(spec, DenseSpec):
            return DenseCoordinateIterator(spec.size)

        if isinstance(spec, CompressedSpec):
            p_begin = storage.pos[parent_pos]
            p_end = storage.pos[parent_pos + 1]

            return CompressedIterator(
                storage.crd,
                int(p_begin),
                int(p_end),
            )

        if isinstance(spec, SingletonSpec):
            return CompressedIterator(
                storage.crd,
                parent_pos,
                parent_pos + 1,
            )

        raise TypeError(f"Unsupported level spec: {type(spec).__name__}")

    @staticmethod
    def make_root_iterator(level) -> SparseIterator:
        if isinstance(level.spec, DenseSpec):
            return IteratorFactory.make_iterator(level.spec, level.storage)
        return IteratorFactory.make_iterator(level.spec, level.storage, parent_pos=0)
