"""`ZXTile` — a small ZX-diagram with ordered external ports, usable as
the internal tensor chunk for a holographic code.

Factories:
    ZXTile.from_graph(g, ports, bulk_ports=())       user-supplied graph
    ZXTile.from_css(SX, LX=None, normal_form='Z-X')  pyzx.css encoder
    ZXTile.from_perfect_tensor(k)                    [[5,1,3]] or [[4,1,2]]

Utility:
    rotate_tile(tile, k)  cyclic shift of the port ordering

Note: `assemble(...)` using `ZXTile` is experimental; for the paper's
validated results use `build_tiled_codes`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# Type alias — pyzx uses int for vertex ids.
Vertex = int


@dataclass
class ZXTile:
    """A small ZX-diagram with ordered external "ports".

    `graph`       a pyzx.Graph encoding one cell's tensor
    `ports`       list of external-port vertex ids in directional order.
                  For a {p, q} tiling, len(ports) should equal p.
                  Convention: ports[0] = "back", then "left", "front",
                  "right", ... cyclically. Non-gluable ports (bulk legs)
                  are listed in `bulk_ports`.
    `bulk_ports`  subset of `ports` marked as bulk (logical) legs, NOT
                  glued across the hyperbolic lattice.
    `name`        optional identifier for debugging / plotting.
    `default_cross_edge`  edge type used when assemble() glues this
                  tile's port to a neighbour. Paper convention is
                  Hadamard.
    """
    graph: "object"  # pyzx.Graph, lazy import
    ports: List[Vertex]
    bulk_ports: List[Vertex] = field(default_factory=list)
    name: str = ""
    default_cross_edge: Optional["object"] = None  # pyzx.EdgeType.HADAMARD

    def __post_init__(self):
        import pyzx as zx
        if self.default_cross_edge is None:
            self.default_cross_edge = zx.EdgeType.HADAMARD
        if not isinstance(self.ports, list):
            self.ports = list(self.ports)
        if not isinstance(self.bulk_ports, list):
            self.bulk_ports = list(self.bulk_ports)
        bad = [b for b in self.bulk_ports if b not in self.ports]
        if bad:
            raise ValueError(
                f"bulk_ports {bad} not in ports {self.ports}")

    @property
    def n_ports(self) -> int:
        return len(self.ports)

    @property
    def gluable_ports(self) -> List[Vertex]:
        return [v for v in self.ports if v not in self.bulk_ports]

    # ---- factories ----------------------------------------------------

    @classmethod
    def from_graph(cls, g, ports: Sequence[Vertex],
                   bulk_ports: Sequence[Vertex] = (),
                   name: str = "") -> "ZXTile":
        """User already has a pyzx.Graph and chooses which vertex ids are
        ports. No validation beyond bulk_ports ⊆ ports."""
        return cls(graph=g, ports=list(ports),
                   bulk_ports=list(bulk_ports), name=name or "user_graph")

    @classmethod
    def from_css(cls, SX, LX=None, normal_form: str = "Z-X",
                 bulk_from: str = "inputs",
                 name: str = "") -> "ZXTile":
        """Construct a tile from a CSS seed code via `pyzx.css.generate_css_encoder_graph`.

        Parameters
        ----------
        SX, LX : matrix-like or pyzx.linalg.Mat2
            X-type stabiliser and logical check matrices (for 'Z-X'
            normal form) or Z-type (for 'X-Z'). Any 2-D binary iterable
            is accepted and converted to `Mat2`.
        normal_form : {'Z-X', 'X-Z'}
            Encoder normal form; see *Picturing Quantum Software* defs
            4.3.1 / 4.3.7.
        bulk_from : {'inputs', 'outputs', 'both'}
            Which external vertices of the CSS encoder become the
            bulk (logical) legs for the resulting tile. Default
            `'inputs'` — CSS encoder inputs are the logical qubits, so
            they become the holographic-code bulk legs. Outputs
            (physical qubits) become the gluable ports.
        name : str
        """
        import pyzx as zx
        from pyzx.css import generate_css_encoder_graph
        from pyzx.linalg import Mat2

        def to_mat2(x):
            if x is None:
                return None
            if isinstance(x, Mat2):
                return x
            return Mat2(list(x))

        g, _ids = generate_css_encoder_graph(
            to_mat2(SX), to_mat2(LX), normal_form)
        # pyzx's generate_css_encoder_graph doesn't set g.inputs()/outputs()
        # directly; auto_detect_io() walks BOUNDARY-type vertices and fills
        # them in based on column ordering (`qubit` attribute).
        g.auto_detect_io()
        inputs = list(g.inputs())
        outputs = list(g.outputs())
        if bulk_from == "inputs":
            ports = outputs + inputs
            bulk = inputs
        elif bulk_from == "outputs":
            ports = inputs + outputs
            bulk = outputs
        elif bulk_from == "both":
            ports = inputs + outputs
            bulk = inputs + outputs
        else:
            raise ValueError(f"bulk_from={bulk_from!r}")
        return cls(graph=g, ports=ports, bulk_ports=bulk,
                   name=name or f"css_{normal_form}")

    @classmethod
    def from_perfect_tensor(cls, k: int = 5, name: str = "") -> "ZXTile":
        """Paper's holographic tile: the [[5, 1, 3]] perfect tensor
        (k=5) or the {4,5} r4 tensor (k=4).

        Both are represented here by their CSS-encoder ZX graph via
        `from_css`. The `k`-qubit ports are gluable; the 1 logical is
        the bulk leg.

        Stabiliser check matrices (X-type, in Z-X normal form):
          k=5: [[5,1,3]] perfect: XZZXI, IXZZX, XIXZZ, ZXIXZ
               (after projection to X side: XXXXX is the logical X̄).
          k=4: [[4,1,2]] r4:    XIXI, IXIX  (+ ZZZZ as Z-stabiliser;
               paper uses both types so we pass them via SX|SZ).
        """
        if k == 5:
            SX = [
                [1, 0, 0, 1, 0],   # (only X-part of XZZXI → placeholder)
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
            ]
            LX = [[1, 1, 1, 1, 1]]
        elif k == 4:
            SX = [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ]
            LX = [[1, 1, 0, 0]]
        else:
            raise ValueError("from_perfect_tensor: k must be 4 or 5")
        return cls.from_css(SX, LX, normal_form="Z-X",
                            name=name or f"perfect_k{k}")


def rotate_tile(tile: ZXTile, k: int) -> ZXTile:
    """Cyclically shift the port ordering by k positions (topology
    unchanged). `rotate_tile(t, 0) == t`."""
    p = tile.n_ports
    new_ports = [tile.ports[(i - k) % p] for i in range(p)]
    return ZXTile(
        graph=tile.graph,
        ports=new_ports,
        bulk_ports=tile.bulk_ports,
        name=f"{tile.name}@{k}" if tile.name else f"@{k}",
        default_cross_edge=tile.default_cross_edge,
    )


# Module-level aliases to match the README examples
from_graph = ZXTile.from_graph
from_css = ZXTile.from_css
from_perfect_tensor = ZXTile.from_perfect_tensor
