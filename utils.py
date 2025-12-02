from backend import xp


def l1_norm(x):
    x = xp.asarray(x)
    return xp.sum(xp.abs(x))


def l2_norm(x):
    x = xp.asarray(x)
    return xp.sqrt(xp.sum(xp.square(x)))