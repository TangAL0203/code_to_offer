import copy

## 实现avg_pool

def avg_pool(data, kernel, stride=1, pad=False, pad_val=0):
    if not data:
        raise ValueError('invalid input')
    N = len(data)
    if N <= kernel:
        if not pad:
            return sum(data) / N
        else:
            return sum([ele for ele in data] + [pad_val] * (kernel - N)) / kernel
    # cal size of returned vector
    # new_n = N - kernel + 1  # stride == 1
    # new_data = []
    # for i in range(new_n):
    #    new_data.append(sum([data[i+j] for j in range(kernel)]) / kernel)
    # return new_data

    # reduce sum operations
    new_n = N - kernel + 1  # stride == 1
    pad_size = 0
    if stride > 1:
        if not pad:
            new_n = (N - (kernel // 2) // stride)
        else:
            new_n = ceil(N - (kernel // 2) / stride)
            pad_size = (ceil(N - (kernel // 2) / stride) - (N - (kernel // 2) / stride)) * stride
    for _ in range(pad_size // 2):
        data.insert(0, pad_val)
    for _ in range(pad_size - (pad_size // 2)):
        data.append(pad_val)

    new_data = []
    cur_sum = 0
    for i in range(0, new_n, stride):
        if i == 0:
            cur_sum = sum([data[i + j] for j in range(kernel)])
        else:
            for g in range(1, stride + 1):
                cur_sum -= data[i - g]
            for g in range(i - stride + kernel, i + kernel):
                cur_sum += data[g]
        new_data.append(cur_sum / kernel)