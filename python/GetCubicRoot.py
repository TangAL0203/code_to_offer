#-*-coding:utf-8-*-
#求一个数的立方根，eps是运行的误差
def part_2(start, end, x, eps):
    mid = (start+end)/2.0
    mid_3 = mid**3
    if abs(mid_3-x)<=eps:
        return mid
    if x>mid_3:
        start = mid+eps
    elif x<mid_3:
        end = mid-eps
    else:
        return mid
    return part_2(start, end, x, eps)

def find_sqrt(x, eps):
    if x>0:
        start = 0
        end = 1e5
    else:
        start = -1e5
        end = 0
    return part_2(start, end, x, eps)

print find_sqrt(100,0.001)