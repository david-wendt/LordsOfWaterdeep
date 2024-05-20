def header(string, width=80):
    length = len(string) + 2
    padding = (width - length) // 2
    top = "#" * (2 * padding + length)
    middle = "=" * padding + " " + string + " " + "=" * padding 
    if 2 * padding + length != width:
        top += "#"
        middle += "="
    return "\n" + top + "\n" + middle + "\n" + top + "\n"