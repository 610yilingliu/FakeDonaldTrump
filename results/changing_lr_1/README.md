```
    if dynamic_lr:
        if epoch == 10:
            lr = 0.01
            change_lr(lr)
        elif epoch == 20:
            lr = 0.005
            change_lr(lr)
        elif epoch == 30:
            lr = 0.002
        elif epoch == 60:
            lr = 0.001
        elif epoch == 100:
            lr = 0.0002
```
