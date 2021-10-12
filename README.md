# SIMD ADD

## Torch 目标 
```python

len  = 30*244*244
a = torch.rand(len, dtype=torch.float)
b = torch.rand(len, dtype=torch.float)
c = torch.rand(len, dtype=torch.float)
d = torch.rand(len, dtype=torch.float)

a.add(b)    # time 约为 1.5ms

c.add_(d)   # time 约为 1ms
```


## Add 目前结果

- ### 4 线程
```
add time = 8708 us 
add_8 time = 8039 us 
vec_add time = 5611 us 
vec_add_2 time = 5689 us 
vec_add_4 time = 5801 us 
```

- ### 8 线程
```
add time = 4536 us 
add_8 time = 3899 us 
vec_add time = 3454 us 
vec_add_2 time = 3074 us 
vec_add_4 time = 3172 us 
```
- ### 12 线程
```
add time = 4054 us 
add_8 time = 3263 us 
vec_add time = 2945 us 
vec_add_2 time = 2934 us 
vec_add_4 time = 2661 us 
```
