from sklearn.preprocessing import StandardScaler


def standard_scaler(data):
    scaler = StandardScaler()
    try:
        scaler.fit(data)
        scaled_data = scaler.transform(data)
    except Exception:
        scaled_data = data
    return scaled_data
