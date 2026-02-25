def close_to_one(x: float, tol: float=1e-3)->bool:
    return abs(x-1.0)<=tol
