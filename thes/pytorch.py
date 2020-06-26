import collections
from torch.utils.data.dataloader import default_collate  # type: ignore


def sequence_batch_collate_v2(batch):
    assert isinstance(batch[0], collections.abc.Sequence), \
            'Only sequences supported'
    # From gunnar code
    transposed = zip(*batch)
    collated = []
    for samples in transposed:
        if isinstance(samples[0], collections.abc.Mapping) \
               and 'do_not_collate' in samples[0]:
            c_samples = samples
        elif getattr(samples[0], 'do_not_collate', False) is True:
            c_samples = samples
        else:
            c_samples = default_collate(samples)
        collated.append(c_samples)
    return collated
