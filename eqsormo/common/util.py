byte_scalars = {
    'TB': 1e12,
    'GB': 1e9,
    'MB': 1e6,
    'KB': 1e3
}

def human_bytes (nbytes):
    for suffix, scalar in byte_scalars.items():
        if nbytes > scalar:
            return f'{nbytes / scalar:.2f} {suffix}'

    return f'{nbytes} bytes'
    
