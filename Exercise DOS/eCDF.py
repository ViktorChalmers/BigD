def eCDF(Mat, q):
    Mat_flatten = Mat.flatten()
    F = 0
    for num in Mat_flatten:
        if num <= q:
            F += 1/len(Mat_flatten)
    return F

