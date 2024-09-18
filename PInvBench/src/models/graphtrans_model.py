from ..models.structgnn_model import StructGNN_Model
from ..modules.graphtrans_module import Struct2Seq


class GraphTrans_Model(StructGNN_Model):
    def __init__(self, args):
        StructGNN_Model.__init__(self, args)
        
        self.model = Struct2Seq(
            num_letters=args.vocab_size,
            node_features=args.hidden,
            edge_features=args.hidden, 
            hidden_dim=args.hidden,
            k_neighbors=args.k_neighbors,
            protein_features=args.features,
            dropout=args.dropout,
            use_mpnn=False)