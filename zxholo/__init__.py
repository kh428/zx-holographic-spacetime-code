"""zxholo — end-to-end pipeline for ZX holographic codes.

Public API (stable):
    # Tile → Lattice → Code
    ZXTile, rotate_tile
    from_graph, from_css, from_perfect_tensor
    build_tiled_codes       (paper-canonical, p4q5 match)
    build_zx_holo_generic   (pentagon-style on arbitrary (p, q))
    assemble                (glue tiles onto a hyperbolic lattice)

    # Extraction (ZX + Pauli webs → symplectic matrices)
    extract_code, col_swap

    # Basis smoothing — paper's random-subsampling heuristic
    smooth_stabiliser_basis, pick_best_smoothed_basis, basis_score

    # Gauge fixing
    apply_gauge, GAUGE_PLUS, GAUGE_ZERO

    # Decoders + channels
    ErasureChannel, PauliChannel, MixedChannel
    make_decoder, run_mc

    # Parallel Monte Carlo (joblib/loky process-pool)
    run_sweep_parallel    (one task per channel; simple)
    run_sweep_chunked     (shot-chunked; better core utilisation)

    # Caching
    save_code, load_code

    # Plotters
    plot_ler_curves, plot_mixed_heatmap

See `examples/` for end-to-end recipes.
"""
from .tile import (                                  # noqa: F401
    ZXTile, rotate_tile,
    from_graph, from_css, from_perfect_tensor,
)
from .lattice import (                                # noqa: F401
    build_tiled_codes, build_zx_holo_generic, col_swap,
)
from .assemble import assemble                        # noqa: F401
from .extract import extract_code                     # noqa: F401
from .smooth import (                                 # noqa: F401
    smooth_stabiliser_basis, pick_best_smoothed_basis, basis_score,
)
from .gauges import apply_gauge, GAUGE_PLUS, GAUGE_ZERO  # noqa: F401
from .decode import (                                 # noqa: F401
    ErasureChannel, PauliChannel, MixedChannel,
    make_decoder, run_mc,
)
from .parallel import (                                # noqa: F401
    run_sweep_parallel, run_sweep_chunked,
)
from .cache import save_code, load_code               # noqa: F401
from .plot import plot_ler_curves, plot_mixed_heatmap  # noqa: F401

__version__ = "0.1.0"
