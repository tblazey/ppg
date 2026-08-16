import json

import nibabel as nib
import numpy as np


def save_nifti(path, data, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.Nifti1Image(data, affine).to_filename(str(path))
    return str(path)


def save_csv(path, time, cnt):
    np.savetxt(path, np.stack((time, cnt), axis=1), delimiter=",", fmt="%.6f")
    return str(path)


def save_pet_json(path, frame_times, duration, radionuclide):
    """Write a minimal BIDS PET JSON sidecar with the given mid-frame times."""

    meta = {
        "FrameTimesStart": (np.asarray(frame_times) - duration / 2.0).tolist(),
        "FrameDuration": [float(duration)] * len(frame_times),
        "TracerRadionuclide": radionuclide,
    }
    with open(path, "w") as json_file:
        json.dump(meta, json_file)
    return str(path)


def replicate_to_4d(cnt, spatial_shape):
    """Broadcast a 1d time series to every voxel of a 4D image."""

    return np.tile(cnt, spatial_shape + (1,))


def replicate_with_noise(cnt, spatial_shape, seed=0, rel_scale=2e-3):
    """Broadcast a 1d time series to every voxel, each with independent noise.

    Voxel-identical copies of the mean curve put per-voxel fits exactly on
    top of a zero-residual point relative to the whole-brain fit, which
    makes L-BFGS-B's finite-difference gradient degenerate. Independent
    per-voxel noise avoids that and is closer to real data anyway.
    """

    n_vox = int(np.prod(spatial_shape))
    rng = np.random.default_rng(seed)
    noisy = cnt[np.newaxis, :] + rng.normal(
        0, rel_scale * cnt.max(), (n_vox, cnt.shape[0])
    )
    return noisy.reshape(spatial_shape + (cnt.shape[0],))
