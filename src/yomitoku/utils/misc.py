def load_charset(charset_path):
    with open(charset_path, "r") as f:
        charset = f.read()
    return charset
