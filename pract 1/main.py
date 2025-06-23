data = [(1, 3), (2, 1), (4, 5), (3, 2)]

sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

print(sorted_data)
