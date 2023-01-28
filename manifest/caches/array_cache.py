"""Array cache."""
from pathlib import Path
from typing import Union

import numpy as np
from sqlitedict import SqliteDict


def open_mmap_arr(file: Union[Path, str], size: float) -> np.memmap:
    """Open memmap."""
    if not Path(file).exists():
        mode = "w+"
    else:
        mode = "r+"
    arr = np.memmap(  # type: ignore
        str(file),
        dtype=np.float32,  # This means we only support float 32
        mode=mode,
        shape=size,
    )
    return arr


class ArrayCache:
    """Array cache."""

    def __init__(self, folder: Union[str, Path]) -> None:
        """
        Initialize the array writer.

        Args:
            folder: folder to write to.
        """
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True, parents=True)
        self.hash2arrloc = SqliteDict(
            self.folder / "hash2arrloc.sqlite", autocommit=True
        )
        # Approx 1GB (I think)
        self.max_memmap_size = 20480000
        self.cur_file_idx = 0
        # Get the last file idx used
        for key in self.hash2arrloc:
            file_data = self.hash2arrloc[key]
            if file_data["file_idx"] > self.cur_file_idx:
                self.cur_file_idx = file_data["file_idx"]
        self.cur_memmap = open_mmap_arr(
            self.folder / f"{self.cur_file_idx}.npy",
            self.max_memmap_size,
        )
        # Make sure there is space left in the memmap
        non_zero = np.nonzero(self.cur_memmap)[0]
        if len(non_zero) > 0:
            self.cur_offset = int(np.max(non_zero) + 1)
        else:
            self.cur_offset = 0
        # If no space, make a new memmap
        if self.cur_offset == self.max_memmap_size:
            self.cur_file_idx += 1
            self.cur_memmap = open_mmap_arr(
                self.folder / f"{self.cur_file_idx}.npy",
                self.max_memmap_size,
            )
            self.cur_offset = 0

    def contains_key(self, key: str) -> bool:
        """
        Check if the key is in the cache.

        Args:
            key: key to check.

        Returns:
            True if the key is in the cache.
        """
        return key in self.hash2arrloc

    def put(self, key: str, arr: np.ndarray) -> None:
        """Save array in store and associate location with key."""
        # Check if there is space in the memmap
        arr_shape = arr.shape
        arr = arr.flatten()
        if len(arr) > self.max_memmap_size:
            raise ValueError(
                f"Array is too large to be cached. Max is {self.max_memmap_size}"
            )
        if self.cur_offset + len(arr) > self.max_memmap_size:
            self.cur_file_idx += 1
            self.cur_memmap = open_mmap_arr(
                self.folder / f"{self.cur_file_idx}.npy",
                self.max_memmap_size,
            )
            self.cur_offset = 0
        self.cur_memmap[self.cur_offset : self.cur_offset + len(arr)] = arr
        self.cur_memmap.flush()
        self.hash2arrloc[key] = {
            "file_idx": self.cur_file_idx,
            "offset": self.cur_offset,
            "flatten_size": len(arr),
            "shape": arr_shape,
            "dtype": arr.dtype,
        }
        self.cur_offset += len(arr)
        return

    def get(self, key: str) -> np.ndarray:
        """Get array associated with location from key."""
        file_data = self.hash2arrloc[key]
        memmap = open_mmap_arr(
            self.folder / f"{file_data['file_idx']}.npy",
            self.max_memmap_size,
        )
        arr = memmap[
            file_data["offset"] : file_data["offset"] + file_data["flatten_size"]
        ]
        return arr.reshape(file_data["shape"]).astype(file_data["dtype"])
