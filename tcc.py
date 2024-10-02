import numpy as np
import pylops
from tqdm import tqdm


def inverting_ricker_butter(amplitude, imp_back, impedance, PPop, PPop_b):
    nx, ny, nz = amplitude.shape()

    # Ricker
    inv_imp_ricker = np.zeros_like(impedance)
    inv_imp_butter = np.zeros_like(impedance)

    for ix in tqdm(range(nx), desc="Processing Ricker traces"):
        for iy in range(ny):
            seismic_trace = amplitude[ix, iy, :]
            impedance_trace = imp_back[ix, iy, :]

            inverted_impedance_trace_ricker = (
                pylops.optimization.leastsquares.normal_equations_inversion(
                    Op=PPop,
                    y=seismic_trace.flatten(),
                    Regs=[],
                    epsI=5e-1,  # tikhonov
                    x0=impedance_trace.flatten(),  # modelo inicial
                    engine="scipy",
                    **dict(maxiter=1000)
                )[0]
            )

            inv_imp_ricker[ix, iy, :] = inverted_impedance_trace_ricker.reshape(nz)

            inverted_impedance_section_butter = (
                pylops.optimization.leastsquares.normal_equations_inversion(
                    Op=PPop_b,
                    y=seismic_trace.flatten(),
                    Regs=[],
                    epsI=5e-1,  # tikhonov
                    x0=impedance_trace.flatten(),  # modelo inicial
                    engine="scipy",
                    **dict(maxiter=1000)
                )[0]
            )

            inv_imp_butter[ix, iy, :] = inverted_impedance_section_butter.reshape(nz)

    inverted_impedance_final_ricker = inv_imp_ricker + imp_back
    inverted_impedance_final_butter = inv_imp_butter + imp_back

    return inverted_impedance_final_ricker, inverted_impedance_final_butter
