from .criterion import Ce
from .criterion import Ce_Dsn
from .criterion import Ce_Dsn_Ohem
from .criterion import Ce_Dsn_Ohem_Single
from .criterion import Ce_Proxy
from .criterion import Ce_Aff
from .criterion import Mse_Aff
from .criterion import Mse_Aff_0
from .criterion import Mse_Aff_01
from .criterion import Mse_Aff_Ohem
from .criterion import Mse_Aff_T
from .criterion import Ce_Edge
from .criterion import Ce_Edge_Parse
from .criterion import Ce_Sphere_Dsn
from .criterion import Ce_Quad
from .criterion import Ce_Triple
from .criterion import Edge_mIOU_Ce_Dsn, Edge_F1_Ce_Dsn

seg_criterion = {
    'ce': Ce,
    'ce_dsn': Ce_Dsn,
    'edge_miou_ce_dsn': Edge_mIOU_Ce_Dsn,
    'edge_f1_ce_dsn': Edge_F1_Ce_Dsn,
    'ce_sphere_dsn': Ce_Sphere_Dsn,
    'ce_dsn_ohem': Ce_Dsn_Ohem,
    'ce_dsn_ohem_single': Ce_Dsn_Ohem_Single,
    'ce_proxy': Ce_Proxy,
    'ce_aff': Ce_Aff,
    'mse_aff': Mse_Aff,
    'mse_aff_0': Mse_Aff_0,
    'mse_aff_01': Mse_Aff_01,
    'mse_aff_ohem': Mse_Aff_Ohem,
    'mse_aff_t': Mse_Aff_T,
    'ce_edge': Ce_Edge,
    'ce_edge_seg': Ce_Edge_Parse,
    'ce_quad': Ce_Quad,
    'ce_triple': Ce_Triple,
}

def get_segmentation_criterion(name, **kwargs):
    return seg_criterion[name.lower()](**kwargs)
