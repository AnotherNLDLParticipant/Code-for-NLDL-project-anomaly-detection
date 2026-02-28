import torch
import torch.nn.functional as F

def anomaly_score(F1, F2, F1hat, F2hat, psi='Euclidean', aggregation='Sum'):
    # Function computing the aggregated anomaly score given extracted features and their reconstructions
    if psi == 'Euclidean':
        dist1 = torch.norm(F1 - F1hat, dim=1)
        dist2 = torch.norm(F2 - F2hat, dim=1)

    elif psi == 'Cosine':
        dist1 = 1 - F.cosine_similarity(F1, F1hat, dim=1)
        dist2 = 1 - F.cosine_similarity(F2, F2hat, dim=1)

    if aggregation == 'Sum':
        return (dist1 + dist2).numpy()
    
    elif aggregation == 'Max':
        return torch.maximum(dist1, dist2).numpy()
    
    elif aggregation == 'Product':
        return (dist1 * dist2).numpy()
    
    elif aggregation == 'First':
        return(dist1.numpy())
    
    elif aggregation == 'Second':
        return(dist2.numpy())