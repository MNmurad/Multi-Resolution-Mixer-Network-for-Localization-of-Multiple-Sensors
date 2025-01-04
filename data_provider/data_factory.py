from data_provider.data_loader import Dataset_Sensor
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Dataset_Sensor

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    data_set = Data(args, flag)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


