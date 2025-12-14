import numpy as np
import scipy.io as sio
from pathlib import Path


def to_mat_compatible(x):
    if x is None:
        return np.nan

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, np.ndarray):
        if x.dtype == object:
            out = np.empty(x.shape, dtype=np.float64)
            it = np.nditer(x, flags=["multi_index", "refs_ok"])
            for v in it:
                val = v.item()
                if val is None:
                    out[it.multi_index] = np.nan
                else:
                    try:
                        out[it.multi_index] = float(val)
                    except Exception:
                        out[it.multi_index] = np.nan
            return out
        else:
            return x

    try:
        return np.array(x, dtype=np.float64)
    except Exception:
        return np.nan


def npz_to_mat(input_dir):
    input_path = Path(input_dir)

    for npz_file in input_path.rglob("*.npz"):
        try:
            data = np.load(npz_file, allow_pickle=True)

            mat_data = {}
            for key in data.files:
                mat_data[key] = to_mat_compatible(data[key])

            mat_file = npz_file.with_suffix(".mat")
            sio.savemat(mat_file, mat_data)

            print(f"Converted: {npz_file} -> {mat_file}")

        except Exception as e:
            print(f"Failed: {npz_file}: {e}")


if __name__ == "__main__":
    directory = "weights/"
    npz_to_mat(directory)
