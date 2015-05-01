"""
Visualize the txt file
"""


def parse_file(filename, pattern):
    """
    Parse the information from the file
    @param filename (str)
    @param pattern (str)
    """
    ret = list()
    with open(filename, 'r') as f:
        line = f.readline()

        while line:
            if line.split()[0] == pattern:
                ret.append(float(line.split()[1]))
            line = f.readline()

    return ret


def output_file(filename, values):
    """
    Write the string to an output_file
    @param filename
    @param values
    """
    with open(filename, 'w') as f:
        for value in values:
            f.write(str(value) + '\n')

if __name__ == '__main__':
    attributes = ['old_time', 'old_speed', 'new_time', 'new_speed']
    for attribute in attributes:
        values = parse_file('timing.txt', attribute)
        output_file(attribute + '.txt', values)
