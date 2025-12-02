from __future__ import annotations

from pathlib import Path

from dataloader import Dataloader
from series_to_numpy import pairs_to_numpy


def main() -> None:
    data_root = Path("data_2")

    loader = Dataloader(data_root)
    collections = loader.available_collections()
    print("Available collections:", collections)

    if not collections:
        return

    collection_name = collections[1]

    x_y_pairing = loader.x_y_pairing(collection_name)
    print("Original x_y_pairing:")
    print(x_y_pairing)

    np_pairing = pairs_to_numpy(x_y_pairing)
    print("\nNumPy-based x_y_pairing:")
    print(np_pairing)


if __name__ == "__main__":
    main()
