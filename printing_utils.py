def header(string, width=80):
    length = len(string) + 2
    padding = (width - length) // 2
    top = "#" * (2 * padding + length)
    middle = "=" * padding + " " + string + " " + "=" * padding 
    if 2 * padding + length != width:
        top += "#"
        middle += "="
    return "\n" + top + "\n" + middle + "\n" + top + "\n"

def pprint(counts: dict[str,int]):
    if not isinstance(counts, dict):
        raise TypeError(f"Unknown input type to pprint: {type(counts)}")
    
    for actionType,count in sorted(counts.items(), reverse=True, key=lambda x: x[1]):
        print(count, '\t', actionType)

if __name__ == '__main__':
    counts = {'a': 5, 'b': 10, 'c': 13, 'd': 7}
    pprint(counts)