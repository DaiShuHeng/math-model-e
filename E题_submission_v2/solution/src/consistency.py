"""Optional clean-to-corrupt consistency; train labels only, no inference overhead."""
import torch
import torch.nn.functional as F


def confident_consistency(clean, corrupt, labels, temperature=2.0, min_confidence=0.6):
    if temperature <= 0 or not 0 <= min_confidence <= 1:
        raise ValueError('Invalid consistency parameters')
    teacher = clean['logits'].detach().float()
    student = corrupt['logits'].float()
    probability = teacher.softmax(-1)
    reliable = (probability.argmax(-1) == labels) & (probability.max(-1).values >= min_confidence)
    per_item = F.kl_div((student / temperature).log_softmax(-1),
                       (teacher / temperature).softmax(-1), reduction='none').sum(-1) * temperature**2
    loss = (per_item * reliable).sum() / reliable.sum().clamp_min(1)
    return loss, reliable.float().mean()
